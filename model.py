import torch
from torch import nn
import torch.nn.functional as F

class mha(nn.Module):
    def __init__(self, n_embd, n_heads, use_bias):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.n_embd = n_embd
        
        # Combine QKV into single linear layer
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=use_bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=use_bias)

    def forward(self, x):
        B, T, C = x.size()
        
        # Single projection for Q, K, V
        q,k,v = self.qkv(x).split(self.n_embd, dim=2)  # (B, S, 3*D)

        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.proj(y)
        return y

class MultiheadGQA(nn.Module):
    def __init__(self, embed_dim, query_heads, kv_heads, use_bias=False): # Default bias to False often
        super().__init__()
        assert embed_dim % query_heads == 0, "embed_dim must be divisible by query_heads"
        assert query_heads % kv_heads == 0, "query_heads must be divisible by kv_heads"

        self.query_heads = query_heads
        self.kv_heads    = kv_heads
        self.head_dim    = embed_dim // query_heads
        self.repeats     = self.query_heads // self.kv_heads
        self.embed_dim   = embed_dim

        self.q_proj   = nn.Linear(embed_dim, query_heads * self.head_dim, bias=use_bias)
        self.k_proj   = nn.Linear(embed_dim, kv_heads * self.head_dim, bias=use_bias)
        self.v_proj   = nn.Linear(embed_dim, kv_heads * self.head_dim, bias=use_bias)

        self.out_proj = nn.Linear(query_heads * self.head_dim, embed_dim, bias=use_bias)

    def forward(self, x):
        B, T, C = x.size()
        assert C == self.embed_dim, f"Input embed_dim ({C}) doesn't match model embed_dim ({self.embed_dim})"

        # Project queries: (B, T, query_heads, head_dim) -> (B, query_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.query_heads, self.head_dim).transpose(1, 2)

        # Project keys and values:
        # Option A: Separate K, V projections
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        # Result shape: (B, kv_heads, T, head_dim)

        # Option B: Original combined projection
        # kv = self.kv_proj(x).view(B, T, 2, self.kv_heads, self.head_dim)
        # k, v = kv.unbind(dim=2) # (B, T, kv_heads, head_dim)
        # k = k.transpose(1, 2) # (B, kv_heads, T, head_dim)
        # v = v.transpose(1, 2) # (B, kv_heads, T, head_dim)

        # --- GQA-specific repetition ---
        # Instead of repeating in memory, SDPA might handle this more efficiently
        # under the hood, especially with torch.compile using Triton kernels.
        # However, the *interface* still requires compatible shapes.
        # Repeat K and V heads without creating large intermediate tensors if possible.
        # This is often handled internally by optimized kernels (like FlashAttention or Triton via compile)
        # but for the standard SDPA, we still need shapes that broadcast correctly or match.
        # Let's explicitly repeat for clarity with standard SDPA, but be aware compile might optimize this away.

        # Efficient repetition for standard SDPA if needed:
        if self.repeats > 1:
             # (B, kv_heads, T, head_dim) -> (B, kv_heads, 1, T, head_dim)
             k = k.unsqueeze(2)
             v = v.unsqueeze(2)
             # -> (B, kv_heads, repeats, T, head_dim)
             k = k.expand(B, self.kv_heads, self.repeats, T, self.head_dim)
             v = v.expand(B, self.kv_heads, self.repeats, T, self.head_dim)
             # -> (B, query_heads, T, head_dim)
             k = k.reshape(B, self.query_heads, T, self.head_dim)
             v = v.reshape(B, self.query_heads, T, self.head_dim)

        # --- Perform scaled dot-product attention ---
        # Input shapes: q(B, Hq, T, D), k(B, Hkv*r, T, D), v(B, Hkv*r, T, D) where Hkv*r = Hq
        # Needs (B, Heads, SeqLen, HeadDim)
        # dropout_p should ideally be self.dropout_p if you add dropout
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        # Output shape: (B, query_heads, T, head_dim)

        # Reshape output: (B, query_heads, T, head_dim) -> (B, T, query_heads, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, self.query_heads * self.head_dim)

        # Final projection
        return self.out_proj(y)

class mlp(nn.Module):
    def __init__(self, n_embd, use_bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4, bias=use_bias),
            nn.GELU(),
            nn.Linear(n_embd*4, n_embd, bias=use_bias)
        )
    
    def forward(self, x):
        return self.net(x)

class transformer_block(nn.Module):
    def __init__(self, n_heads, n_embd, use_bias):
        super().__init__()
        self.attn = MultiheadGQA(n_embd, n_heads, 2, use_bias)
        #self.attn = mha(n_embd, n_heads, use_bias)
        self.mlp = mlp(n_embd, use_bias)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class llm(nn.Module):
    def __init__(self, vocab_size, n_heads, n_embd, n_layers, use_bias, seq_len, yap=True):
        super().__init__()
        # save init params
        self.model_params = {
            'vocab_size': vocab_size,
            'n_heads': n_heads,
            'n_embd': n_embd,
            'n_layers': n_layers,
            'use_bias': use_bias,
            'seq_len': seq_len
        }
        
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.ModuleList([
            transformer_block(n_heads, n_embd, use_bias) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=use_bias)
        
        if yap:
            # parameter calculation
            embed_params = vocab_size * n_embd
            attn_params = 4 * n_embd * n_embd
            mlp_params = n_embd * 4 + n_embd * 4
            ln_params = 2 * n_embd
            one_layer = attn_params + mlp_params + 2 * ln_params
            total_params = embed_params + n_layers * one_layer + n_embd * vocab_size + ln_params
            print("----- params -----")
            print(f"embedding params: {embed_params:,}")
            print(f"attention params per layer: {attn_params:,}")
            print(f"mlp params per layer: {mlp_params:,}")
            print(f"one transformer layer: {one_layer:,}")
            print(f"total params: {total_params:,}")
            print("----- ------ -----")

    def forward(self, x, padding_mask=None):
        # Get the actual sequence length from the input
        batch_size, seq_length = x.size()
        
        # Ensure sequence length doesn't exceed max seq_len
        if seq_length > self.seq_len:
            raise ValueError(f"Input sequence length ({seq_length}) exceeds maximum seq_len ({self.seq_len})")
        
        # Generate positional embeddings only for the actual sequence length
        positions = torch.arange(seq_length, device=x.device)
        x = self.embed(x) + self.pos_embed(positions)  # Dynamically size pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        return self.head(x)

    def generate(self, x, max_tokens=10, temperature=1.0, min_p=0.05, echo_back=False):
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self(x)[:, -1, :]  # Get logits for the last token
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                probs = F.softmax(logits, dim=-1)
                
                if min_p > 0.0:
                    max_p = torch.max(probs)
                    probs[probs < min_p * max_p] = 0
                    probs = probs / probs.sum()
                
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)
        
        x = x.tolist()[0]
        if not echo_back:
            x = x[-max_tokens:]
        return x
        