import torch
from torch import nn
import torch.nn.functional as F

class mha(nn.Module):
    def __init__(self, d_model, n_heads, use_bias):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Combine QKV into single linear layer
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.proj = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, x):
        B, T, C = x.size()
        
        # Single projection for Q, K, V
        q,k,v = self.qkv(x).split(self.d_model, dim=2)  # (B, S, 3*D)

        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.proj(y)
        return y
    
class mlp(nn.Module):
    def __init__(self, d_model, n_embd, use_bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, n_embd, bias=use_bias),
            nn.GELU(),
            nn.Linear(n_embd, d_model, bias=use_bias)
        )
    
    def forward(self, x):
        return self.net(x)

class transformer_block(nn.Module):
    def __init__(self, d_model, n_heads, n_embd, use_bias):
        super().__init__()
        self.attn = mha(d_model, n_heads, use_bias)
        self.mlp = mlp(d_model, n_embd, use_bias)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class llm(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_embd, n_layers, use_bias, seq_len, yap=True):
        super().__init__()
        # save init params
        self.model_params = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_embd': n_embd,
            'n_layers': n_layers,
            'use_bias': use_bias,
            'seq_len': seq_len
        }
        
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            transformer_block(d_model, n_heads, n_embd, use_bias) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=use_bias)
        
        if yap:
            # parameter calculation
            embed_params = vocab_size * d_model
            attn_params = 4 * d_model * d_model
            mlp_params = d_model * n_embd + n_embd * d_model
            ln_params = 2 * d_model
            one_layer = attn_params + mlp_params + 2 * ln_params
            total_params = embed_params + n_layers * one_layer + d_model * vocab_size + ln_params
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