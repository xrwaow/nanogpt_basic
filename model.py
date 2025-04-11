# model.py
import torch
from torch import nn
import torch.nn.functional as F
import math

# --- RMSNorm Layer ---
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter (learnable gain)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate the Root Mean Square: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # Normalize: x / rms
        return x / rms

    def forward(self, x):
        # Normalize the input and scale by the learnable weight parameter
        output = self._norm(x.float()).type_as(x) # Cast to float for calculation, then back
        return output * self.weight


# Helper function for Rotary Positional Embeddings (RoPE)
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
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
    # Note: This class still reads from config if used.
    # Needs modification to accept params if intended for use.
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
    def __init__(self, n_embd, n_heads, kv_heads, use_bias, rope_theta, block_size):
        super().__init__()
        assert n_embd % n_heads == 0, "N_EMBD must be divisible by N_HEADS"
        assert n_heads % kv_heads == 0, "N_HEADS must be divisible by KV_HEADS"

        self.query_heads = n_heads
        self.kv_heads    = kv_heads
        self.head_dim    = n_embd // n_heads
        self.repeats     = self.query_heads // self.kv_heads
        self.embed_dim   = n_embd
        self.max_seq_len = block_size
        self.rope_theta  = rope_theta

        # Projections for Q, K, V
        self.q_proj   = nn.Linear(self.embed_dim, self.query_heads * self.head_dim, bias=use_bias)
        self.k_proj   = nn.Linear(self.embed_dim, self.kv_heads    * self.head_dim, bias=use_bias)
        self.v_proj   = nn.Linear(self.embed_dim, self.kv_heads    * self.head_dim, bias=use_bias)

        # Output projection
        self.out_proj = nn.Linear(self.query_heads * self.head_dim, self.embed_dim, bias=use_bias)

        # Precompute RoPE frequencies
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

        # Repeat K and V heads if necessary (if repeats > 1)
        k = self.repeat_kv(k, self.repeats) # (B, Hkv, T, D) -> (B, Hq, T, D)
        v = self.repeat_kv(v, self.repeats) # (B, Hkv, T, D) -> (B, Hq, T, D)

        # Perform scaled dot-product attention
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True # is_causal=True enables causal mask
        )
        # Output shape: (B, query_heads, T, head_dim)

        # (B, Hq, T, D) -> (B, T, Hq, D) -> (B, T, Hq*D = C)
        y = y.transpose(1, 2).contiguous().view(B, T, self.query_heads * self.head_dim)
        return self.out_proj(y) # (B, T, C)


class mlp_depricated(nn.Module):
    def __init__(self, n_embd, use_bias):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_embd*4, bias=use_bias)
        self.act = nn.GELU() # Consider SiLU/SwiGLU for modern LLMs if desired
        self.fc2 = nn.Linear(n_embd*4, n_embd, bias=use_bias)

        self.net = nn.Sequential(self.fc1, self.act, self.fc2)

    def forward(self, x):
        return self.net(x)

#class SwiGLU(nn.Module):
#    def forward(self, x):
#        x, gate = x.chunk(2, dim=-1)
#       return x * F.silu(gate)

class MLP(nn.Module):
    def __init__(self, n_embd, use_bias):
        super().__init__()
        hidden_dim = n_embd * 4

        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=use_bias)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=use_bias)
        #self.act_fn = nn.SiLU() # Swish activation

    def forward(self, x):
        # F.silu(gate_proj(x)) * up_proj(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        #return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class transformer_block(nn.Module):
    def __init__(self, n_embd, n_heads, kv_heads, use_bias, rope_theta, block_size):
        super().__init__()
        self.attn = MultiheadGQA(n_embd, n_heads, kv_heads, use_bias, rope_theta, block_size)
        self.mlp = MLP(n_embd, use_bias)
        # Use RMSNorm instead of LayerNorm
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class llm(nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, n_embd, kv_heads, use_bias, rope_theta, block_size, tie_weights=True, print_model_params=False):
        super().__init__()

        self.model_params = {
            'vocab_size': vocab_size,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_embd': n_embd,
            'kv_heads': kv_heads,
            'use_bias': use_bias,
            'rope_theta': rope_theta,
            'block_size': block_size,
            'tie_weights': tie_weights, # Store tie_weights config
            'print_model_params': print_model_params
        }

        self.seq_len = block_size # Use passed block_size
        self.embed = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList([
            transformer_block(n_embd, n_heads, kv_heads, use_bias, rope_theta, block_size)
            for _ in range(n_layers)
        ])
        # Use RMSNorm for final normalization
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=use_bias) # Output head

        # Initialize embedding weights (always happens)
        init_std_embed = (n_embd) ** -0.5
        torch.nn.init.normal_(self.embed.weight, mean=0.0, std=init_std_embed)

        if tie_weights:
            self.head.weight = self.embed.weight # Weight tying
            # Re-initialize tied weights (using the same std as embedding)
            if print_model_params:
                print(f"Tying weights and re-initializing tied head.weight with std={init_std_embed:.4f}")
            # No need to re-init embed.weight again, it's already done
        else:
            # Initialize head weights separately if not tied
            init_std_head = (n_embd) ** -0.5 # Or use a different initialization if desired
            torch.nn.init.normal_(self.head.weight, mean=0.0, std=init_std_head)
            if print_model_params:
                print(f"NOT tying weights. Initializing head.weight separately with std={init_std_head:.4f}")

        # Initialize head bias if it exists and is not tied to embedding bias (which doesn't exist)
        if use_bias and self.head.bias is not None:
             torch.nn.init.zeros_(self.head.bias)

        if print_model_params:
            self._print_params() # Call helper for parameter printing

    def _print_params(self):
        """Helper function to calculate and print parameter counts."""
        print("----- Parameter Calculation (RMSNorm, RoPE, GQA) -----")

        def count_params(module):
            # Count only trainable parameters
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        # Parameters for Embedding layer
        embed_params = count_params(self.embed)
        n_layers_actual = len(self.blocks) # Get actual number of layers

        # Parameters for one block (use the first block)
        if self.blocks: # Ensure blocks list is not empty
            first_block = self.blocks[0]
            # Note: RoPE freqs are in buffers, not parameters, so not counted here
            attn_params_per_layer = count_params(first_block.attn)
            mlp_params_per_layer = count_params(first_block.mlp)
            # RMSNorm only has 'weight' parameter per instance
            norm_params_per_layer = count_params(first_block.ln1) + count_params(first_block.ln2)
            one_layer_params = count_params(first_block)
        else:
            attn_params_per_layer = mlp_params_per_layer = norm_params_per_layer = one_layer_params = 0

        # Parameters for final layers
        ln_f_params = count_params(self.ln_f)

        # Head params (consider weight tying)
        tie_weights = self.model_params.get('tie_weights', True) # Get from stored params
        if tie_weights:
            # Only count bias if it exists and is trainable
            head_params_effective = count_params(self.head.bias) if self.head.bias is not None else 0
            tying_info = "(Tied Weight)"
        else:
            head_params_effective = count_params(self.head) # Count weight and bias (if exists)
            tying_info = "(Untied)"

        # Calculate total directly from the model (most reliable)
        total_params_actual = count_params(self)

        print(f"Token Embedding: {embed_params:,}")
        if self.blocks:
            print("--- Per Layer ---")
            print(f"  Attention (GQA + RoPE): {attn_params_per_layer:,} (RoPE freqs are non-parameter buffers)")
            print(f"  MLP (SwiGLU): {mlp_params_per_layer:,}")
            print(f"  RMSNorms (x2): {norm_params_per_layer:,}")
            print(f"  Total per Transformer Layer: {one_layer_params:,}")
            print(f"Total for {n_layers_actual} Layers: {one_layer_params * n_layers_actual:,}")
        print("--- Final Layers ---")
        print(f"Final RMSNorm: {ln_f_params:,}")
        print(f"Output Head {tying_info}: {head_params_effective:,}")
        print("--- Grand Total ---")
        print(f"Total Trainable Parameters: {total_params_actual:,}")
        print("----- --------------------------------------------- -----")


    def forward(self, x):
        batch_size, seq_length = x.size()

        if seq_length > self.seq_len:
             raise ValueError(f"Input sequence length ({seq_length}) exceeds model's maximum sequence length ({self.seq_len}) required for RoPE.")

        x = self.embed(x) # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad() # Decorator for no_grad context
    def generate(self, x, max_tokens=10, temperature=1.0, top_k=None, echo_back=False):
        """
        Generates token sequences with optional sampling strategies.
        Uses standard Top-K sampling.
        """

        generated_tokens = [] # Store only newly generated tokens
        current_context = x # Start with the initial context (B, T_prompt)

        for _ in range(max_tokens):
            # Ensure context doesn't exceed model's max sequence length for the *next* forward pass
            current_context_trimmed = current_context[:, -self.seq_len:] # Use self.seq_len set during init

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

            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append the new token to the context for the next step
            current_context = torch.cat([current_context, next_token], dim=1) # (B, T_prompt + generated_len)
            generated_tokens.append(next_token)

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