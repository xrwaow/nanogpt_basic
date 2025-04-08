import torch
from torch import nn
import torch.nn.functional as F
import math

# Helper function for Rotary Positional Embeddings (RoPE)
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequency and complex number representations for RoPE.

    Args:
        dim (int): The dimension of the embeddings (specifically, head dimension).
        end (int): The maximum sequence length.
        theta (float): The base frequency for RoPE.

    Returns:
        torch.Tensor: A tensor of complex numbers shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # Shape: (dim / 2)
    t = torch.arange(end, device=freqs.device) # Shape: (end)
    freqs = torch.outer(t, freqs) # Shape: (end, dim / 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # Shape: (end, dim / 2)
    return freqs_cis

# Helper function to apply RoPE
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embeddings to query (xq) and key (xk) tensors.

    Args:
        xq (torch.Tensor): Query tensor with shape (B, H, T, D_head).
        xk (torch.Tensor): Key tensor with shape (B, H, T, D_head).
        freqs_cis (torch.Tensor): Precomputed complex numbers for RoPE
                                  shape (T, D_head // 2) or broadcastable.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Query and Key tensors with RoPE applied.
                                           Returned tensors will have the same dtype as inputs xq, xk.
    """
    # Store original dtype
    input_dtype = xq.dtype

    # Reshape xq and xk to view pairs of dimensions as complex numbers
    # Cast to float32 for complex math precision
    # (B, H, T, D_head) -> (B, H, T, D_head/2, 2)
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Convert to complex numbers: x = xr + i*xi
    # Shape: (B, H, T, D_head/2)
    xq = torch.view_as_complex(torch.stack([xq_r, xq_i], dim=-1)) # complex64
    xk = torch.view_as_complex(torch.stack([xk_r, xk_i], dim=-1)) # complex64

    # Reshape freqs_cis for broadcasting: (T, D_head/2) -> (1, 1, T, D_head/2)
    # Ensure freqs_cis is on the same device as xq
    freqs_cis = freqs_cis.to(xq.device)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1)

    # Apply rotation by complex multiplication: x * freqs_cis
    # Shape: (B, H, T, D_head/2) * (1, 1, T, D_head/2) -> (B, H, T, D_head/2)
    xq_out = torch.view_as_real(xq * freqs_cis).flatten(3) # float32
    xk_out = torch.view_as_real(xk * freqs_cis).flatten(3) # float32

    # Cast back to the original input dtype
    return xq_out.to(input_dtype), xk_out.to(input_dtype)


# Keep mha class definition if you might want to switch back, but it's unused with GQA+RoPE
class mha(nn.Module):
    def __init__(self, n_embd, n_heads, use_bias):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"

        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // n_heads # Define head_dim for clarity

        # Combine QKV into single linear layer
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=use_bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=use_bias)

    def forward(self, x):
        B, T, C = x.size()
        assert C == self.n_embd, f"Input embedding dimension ({C}) doesn't match model ({self.n_embd})"

        # Single projection for Q, K, V
        q,k,v = self.qkv(x).split(self.n_embd, dim=2)  # (B, T, D) -> 3 * (B, T, D)

        # Reshape and transpose for multi-head attention
        # (B, T, D) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Use scaled dot product attention (efficient implementation)
        # is_causal=True handles the masking automatically
        # RoPE would be applied here to q and k if this class used it
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        # Re-assemble head outputs
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y) # Final output projection
        return y

class MultiheadGQA(nn.Module):
    def __init__(self, embed_dim, query_heads, kv_heads, max_seq_len, use_bias=False, rope_theta=10000.0):
        super().__init__()
        assert embed_dim % query_heads == 0, "embed_dim must be divisible by query_heads"
        assert query_heads % kv_heads == 0, "query_heads must be divisible by kv_heads"

        self.query_heads = query_heads
        self.kv_heads    = kv_heads
        self.head_dim    = embed_dim // query_heads
        self.repeats     = self.query_heads // self.kv_heads
        self.embed_dim   = embed_dim

        # Projections for Q, K, V
        self.q_proj   = nn.Linear(embed_dim, query_heads * self.head_dim, bias=use_bias)
        self.k_proj   = nn.Linear(embed_dim, kv_heads    * self.head_dim, bias=use_bias)
        self.v_proj   = nn.Linear(embed_dim, kv_heads    * self.head_dim, bias=use_bias)

        # Output projection
        self.out_proj = nn.Linear(query_heads * self.head_dim, embed_dim, bias=use_bias)

        # Precompute RoPE frequencies
        # Needs head_dim and max_seq_len
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        # Compute frequencies on CPU initially, will be moved to device in forward
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2, self.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False) # Register as buffer, not parameter

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        B, n_kv_heads, T, head_dim = x.shape
        if n_rep == 1:
            return x
        # (B, N_KV_HEADS, T, HEAD_DIM) -> (B, N_KV_HEADS, 1, T, HEAD_DIM)
        x = x.unsqueeze(2)
        # -> (B, N_KV_HEADS, N_REP, T, HEAD_DIM)
        x = x.expand(B, n_kv_heads, n_rep, T, head_dim)
        # -> (B, N_KV_HEADS * N_REP, T, HEAD_DIM)
        # Effectively reshaping to (B, N_Q_HEADS, T, HEAD_DIM)
        return x.reshape(B, n_kv_heads * n_rep, T, head_dim)

    def forward(self, x):
        B, T, C = x.size()
        assert C == self.embed_dim, f"Input embed_dim ({C}) doesn't match model embed_dim ({self.embed_dim})"
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds model maximum {self.max_seq_len}"


        # Project Q, K, V
        q = self.q_proj(x) # (B, T, Hq * D)
        k = self.k_proj(x) # (B, T, Hkv * D)
        v = self.v_proj(x) # (B, T, Hkv * D)

        # Reshape Q, K, V for attention
        # (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.query_heads, self.head_dim).transpose(1, 2) # (B, Hq, T, D)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)   # (B, Hkv, T, D)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)   # (B, Hkv, T, D)

        # Apply RoPE to Q and K *before* repeating K/V
        # Slice precomputed freqs_cis based on current sequence length T
        # Ensure freqs_cis is on the correct device
        current_freqs_cis = self.freqs_cis[:T].to(q.device)

        # Apply rotary embeddings. Function ensures output dtype matches input dtype.
        q, k = apply_rotary_emb(q, k, freqs_cis=current_freqs_cis)
        # q shape: (B, Hq, T, D), dtype should match original q (e.g., bfloat16)
        # k shape: (B, Hkv, T, D), dtype should match original k (e.g., bfloat16)

        # *** Check dtypes before SDPA ***
        # This check is crucial because SDPA requires Q, K, V to have the same dtype.
        # Although apply_rotary_emb *should* return the correct dtype, this makes it explicit.
        target_dtype = v.dtype # Usually bfloat16 or float16
        if q.dtype != target_dtype:
            q = q.to(target_dtype)
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        # Now q, k, v all have dtype `target_dtype`

        # Repeat K and V heads if necessary (if repeats > 1)
        k = self.repeat_kv(k, self.repeats) # (B, Hkv, T, D) -> (B, Hq, T, D)
        v = self.repeat_kv(v, self.repeats) # (B, Hkv, T, D) -> (B, Hq, T, D)

        # Perform scaled dot-product attention
        # Inputs q, k, v should now all have the same dtype (target_dtype)
        # and shape (B, Hq, T, D) after repeat_kv
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True # is_causal=True enables causal mask
        )
        # Output shape: (B, query_heads, T, head_dim)

        # Reshape output and project
        # (B, Hq, T, D) -> (B, T, Hq, D) -> (B, T, Hq*D = C)
        y = y.transpose(1, 2).contiguous().view(B, T, self.query_heads * self.head_dim)
        return self.out_proj(y) # (B, T, C)


class mlp(nn.Module):
    def __init__(self, n_embd, use_bias):
        super().__init__()
        # Standard MLP structure: Linear -> Activation -> Linear
        self.fc1 = nn.Linear(n_embd, n_embd*4, bias=use_bias)
        self.act = nn.GELU() # Consider SiLU/SwiGLU for modern LLMs if desired
        self.fc2 = nn.Linear(n_embd*4, n_embd, bias=use_bias)
        # Combine into sequential for forward pass simplicity
        self.net = nn.Sequential(self.fc1, self.act, self.fc2)

    def forward(self, x):
        return self.net(x)

class transformer_block(nn.Module):
    def __init__(self, n_heads, n_embd, use_bias, kv_heads, max_seq_len, rope_theta=10000.0): # Pass max_seq_len and theta
        super().__init__()
        # Use Grouped Query Attention with RoPE
        self.attn = MultiheadGQA(n_embd, n_heads, kv_heads, max_seq_len, use_bias, rope_theta)
        self.mlp = mlp(n_embd, use_bias)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LayerNorm structure
        x = x + self.attn(self.ln1(x)) # Add residual connection after attention
        x = x + self.mlp(self.ln2(x))  # Add residual connection after MLP
        return x

class llm(nn.Module):
    def __init__(self, vocab_size, n_heads, n_embd, n_layers, seq_len, use_bias=False, kv_heads=2, rope_theta=10000.0, yap=True): # Added rope_theta
        super().__init__()
        # save init params
        self.model_params = {
            'vocab_size': vocab_size,
            'n_heads': n_heads,
            'n_embd': n_embd,
            'n_layers': n_layers,
            'seq_len': seq_len, # Maximum sequence length the model was trained with / can handle
            'use_bias': use_bias,
            'kv_heads': kv_heads,
            'rope_theta': rope_theta # Store RoPE base frequency
        }

        self.seq_len = seq_len # Store max sequence length
        self.embed = nn.Embedding(vocab_size, n_embd)
        # Positional embeddings are now handled by RoPE within the attention mechanism

        # Pass kv_heads, max_seq_len, and rope_theta to transformer_block
        self.blocks = nn.ModuleList([
            transformer_block(n_heads, n_embd, use_bias, kv_heads=kv_heads, max_seq_len=seq_len, rope_theta=rope_theta)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd) # Final LayerNorm
        self.head = nn.Linear(n_embd, vocab_size, bias=use_bias) # Output head

        self.head.weight = self.embed.weight# - shit loss with this

        init_std = (n_embd) ** -0.5
        print(f"Re-initializing tied weights with std={init_std:.4f} (1/sqrt(n_embd))")
        torch.nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)

        if yap:
            self._print_params() # Call helper for parameter printing

    def _print_params(self):
        """Helper function to calculate and print parameter counts."""
        print("----- Parameter Calculation (RoPE) -----")

        def count_params(module):
            # Count only trainable parameters
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        embed_params = count_params(self.embed)

        # Parameters for one block (use the first block)
        if self.blocks: # Ensure blocks list is not empty
            first_block = self.blocks[0]
            # Note: RoPE freqs are in buffers, not parameters, so not counted here
            attn_params_per_layer = count_params(first_block.attn)
            mlp_params_per_layer = count_params(first_block.mlp)
            ln_params_per_layer = count_params(first_block.ln1) + count_params(first_block.ln2)
            one_layer_params = count_params(first_block)
        else:
            attn_params_per_layer = mlp_params_per_layer = ln_params_per_layer = one_layer_params = 0

        # Parameters for final layers
        ln_f_params = count_params(self.ln_f)
        # Head params (consider weight tying)
        if hasattr(self.head, 'weight') and hasattr(self.embed, 'weight') and self.head.weight is self.embed.weight:
            # Only count bias if it exists and is trainable
            head_params_effective = count_params(self.head.bias) if self.head.bias is not None else 0
            tying_info = "(Tied Weight)"
        else:
            head_params_effective = count_params(self.head)
            tying_info = "(Untied)"


        # Calculate total directly from the model (most reliable)
        total_params_actual = count_params(self)

        print(f"Token Embedding: {embed_params:,}")
        if self.blocks:
            print("--- Per Layer ---")
            print(f"  Attention (GQA + RoPE): {attn_params_per_layer:,} (RoPE freqs are non-parameter buffers)")
            print(f"  MLP: {mlp_params_per_layer:,}")
            print(f"  LayerNorms (x2): {ln_params_per_layer:,}")
            print(f"  Total per Transformer Layer: {one_layer_params:,}")
            print(f"Total for {len(self.blocks)} Layers: {one_layer_params * len(self.blocks):,}")
        print("--- Final Layers ---")
        print(f"Final LayerNorm: {ln_f_params:,}")
        print(f"Output Head {tying_info}: {head_params_effective:,}")
        print("--- Grand Total ---")
        print(f"Total Trainable Parameters: {total_params_actual:,}")
        print("----- ------------------------------ -----")


    def forward(self, x):
        # Get the actual sequence length from the input
        batch_size, seq_length = x.size()

        # Input validation: Check if sequence length exceeds max_seq_len
        if seq_length > self.seq_len:
             raise ValueError(f"Input sequence length ({seq_length}) exceeds model's maximum sequence length ({self.seq_len}) required for RoPE.")

        # Get token embeddings
        x = self.embed(x) # (B, T, C)

        # RoPE is applied *inside* each transformer block's attention layer

        # Apply transformer blocks sequentially
        for block in self.blocks:
            x = block(x)

        # Apply final layer normalization
        x = self.ln_f(x)
        # Final linear layer (head) to get logits
        return self.head(x)

    @torch.no_grad() # Decorator for no_grad context
    def generate(self, x, max_tokens=10, temperature=1.0, min_p=0.05, top_k=None, echo_back=False):
        """
        Generates token sequences with optional sampling strategies.
        Uses standard Top-P (nucleus) sampling if min_p is between 0 and 1.
        """
        self.eval() # Set model to evaluation mode

        generated_tokens = [] # Store only newly generated tokens
        current_context = x # Start with the initial context (B, T_prompt)

        for _ in range(max_tokens):
            # Ensure context doesn't exceed model's max sequence length for the *next* forward pass
            current_context_trimmed = current_context[:, -self.seq_len:]

            # Get logits for the very last token prediction
            logits = self(current_context_trimmed)[:, -1, :] # (B, VocabSize)

            # Apply temperature scaling
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # Optional Top-K sampling
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # Mask out logits below k-th value

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1) # (B, VocabSize)

            # Optional Nucleus (Top-P) sampling
            if 0 < min_p < 1.0:
                 probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                 probs_sum = torch.cumsum(probs_sort, dim=-1)
                 # Remove tokens with cumulative probability above the threshold (token *after* the threshold is crossed)
                 mask = (probs_sum - probs_sort) > min_p
                 probs_sort[mask] = 0.0 # Zero out probabilities below the threshold
                 probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # Re-normalize
                 # Sample from the modified distribution
                 next_token_idx = torch.multinomial(probs_sort, num_samples=1) # Get index in sorted list
                 next_token = torch.gather(probs_idx, -1, next_token_idx) # Get original token ID
            else: # Simple multinomial sampling (works after Top-K modifications too)
                 next_token = torch.multinomial(probs, num_samples=1) # (B, 1)


            # Append the new token to the context for the next step
            current_context = torch.cat([current_context, next_token], dim=1) # (B, T_prompt + generated_len)
            generated_tokens.append(next_token)

        # Set model back to training mode if needed elsewhere (optional)
        # self.train()

        # Concatenate all generated tokens
        if generated_tokens:
            # generated_sequence_tensor shape (B, max_tokens)
            generated_sequence_tensor = torch.cat(generated_tokens, dim=1)
            # Assuming batch size B=1 for typical generation prompts
            generated_sequence = generated_sequence_tensor.squeeze(0).tolist() # (max_tokens) list
        else:
            generated_sequence = []

        # Return either just the generated part or the full sequence
        if echo_back:
             # Assuming batch size B=1
             return current_context.squeeze(0).tolist() # Return full context including prompt
        else:
             return generated_sequence # Return only the newly generated tokens