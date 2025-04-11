import torch
from transformers import AutoTokenizer
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import math
import os
import functools
import json
import inspect

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

import config as cfg
from load_data import *
from model import llm

torch.manual_seed(cfg.SEED)

checkpoint_base_dir = os.path.join(cfg.CHECKPOINT_DIR, cfg.SAVE_DIR_NAME)
os.makedirs(checkpoint_base_dir, exist_ok=True)
config_save_path = os.path.join(checkpoint_base_dir, "config.json")

config_dict = {}
for name, attr in inspect.getmembers(cfg):
    # Exclude built-in attributes and modules
    if not name.startswith("__") and not inspect.ismodule(attr):
        # Convert torch.dtype to string for JSON serialization
        if isinstance(attr, torch.dtype):
            config_dict[name] = str(attr)
        else:
            config_dict[name] = attr
try:
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
except Exception as e:
    print(f"Error saving config.json: {e}")

current_macro_batch_layer_sparsity = {}

def sparsity_hook_factory(layer_idx):
    """Creates a hook function that knows its layer index."""
    def sparsity_hook(module, input, output): # Keep signature standard
        """Forward hook to calculate activation sparsity based on the INPUT to a module."""
        # We hook the input to the MLP's down_proj layer.
        # 'input' is a tuple containing the input tensors. We need the first one.
        if not input or not isinstance(input[0], torch.Tensor):
             # Skip if input is unexpected
             return
        
        target_tensor = input[0]
        threshold = 1e-3 # Define the threshold for 'zero' activation
        is_zero = torch.abs(target_tensor) < threshold
        sparsity = torch.mean(is_zero.float()).item()
        # Append to the list for the specific layer within the current macro batch
        if layer_idx not in current_macro_batch_layer_sparsity:
             current_macro_batch_layer_sparsity[layer_idx] = []
        current_macro_batch_layer_sparsity[layer_idx].append(sparsity)
    return sparsity_hook

def compute_grad_norm(model):
    total_norm = 0.0 # Use float for accumulation
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5 # Calculate sqrt at the end
    return total_norm

def get_lr(it):
    # Uses learning_rate, warmup_iters, lr_decay_iters, min_lr from config
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

data_path = prepare_data_optimized(cfg.dataset_name, cfg.MAX_TOKENS, tokenizer)
data = torch.load(data_path, map_location='cpu', weights_only=False)
if data.device != torch.device('cpu'):
    data = data.to('cpu')
if data.dtype != torch.long:
    data = data.long()

# create model using vocab_size and parameters from config
llm_model = llm(
    vocab_size=tokenizer.vocab_size,
    n_layers=cfg.N_LAYERS,
    n_heads=cfg.N_HEADS,
    n_embd=cfg.N_EMBD,
    kv_heads=cfg.KV_HEADS,
    use_bias=cfg.USE_BIAS,
    rope_theta=cfg.ROPE_THETA,
    block_size=cfg.BLOCK_SIZE,
    print_model_params=cfg.PRINT_MODEL_PARAMS
)

hook_handles = []

if hasattr(llm_model, 'blocks') and isinstance(llm_model.blocks, torch.nn.ModuleList):
    for i, block in enumerate(llm_model.blocks):
        try:
            # Target the INPUT of the down_proj layer in the MLP block
            # This captures the state after the SiLU activation and multiplication
            target_module = block.mlp.down_proj
            hook_func = sparsity_hook_factory(i)
            # Use register_forward_hook, but the hook function itself looks at the 'input' argument
            handle = target_module.register_forward_hook(hook_func)
            hook_handles.append(handle)
            if cfg.PRINT_MODEL_PARAMS: # Only print if enabled
                print(f"Registered sparsity hook on input of Layer {i} MLP down_proj.")
        except AttributeError:
             print(f"Warning: Could not find mlp.down_proj in block {i}. Skipping hook.")
        except Exception as e:
            print(f"Warning: Could not register sparsity hook for layer {i}: {e}")
else:
    print("Warning: Model does not have 'blocks' attribute or it's not a ModuleList. No sparsity hooks registered.")

llm_model = llm_model.to(cfg.DEVICE)
llm_model = torch.compile(llm_model)

optimizer = torch.optim.AdamW(
    llm_model.parameters(),
    lr=cfg.learning_rate,
    betas=(cfg.beta1, cfg.beta2),
    weight_decay=cfg.weight_decay,
    fused=(cfg.DEVICE=="cuda")
)

criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler(enabled=(cfg.DEVICE == "cuda"))

total_tokens = 0

all_metrics = {
    'loss': [],
    'grad_norm': [],
    'lr': [],
    'layer_sparsity': {layer_idx: [] for layer_idx in range(cfg.N_LAYERS)} # Store list for each layer
}
# Use MAX_TOKENS from config for progress bar total
progress_bar = tqdm(total=cfg.MAX_TOKENS, desc='training progress', unit='tok')
iter_num = 0

# training loop
try: # Wrap training loop for cleanup
    while total_tokens < cfg.MAX_TOKENS:
        # determine and set the learning rate for this iteration using decay_lr from config
        lr = get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Reset per-layer sparsity list for the *macro* batch
        current_macro_batch_layer_sparsity.clear()

        optimizer.zero_grad(set_to_none=True) # More memory efficient

        # Use GRAD_ACCUM_STEPS from config
        for micro_step in range(cfg.GRAD_ACCUM_STEPS):
            # Use DEVICE and AMP_DTYPE from config for autocast
            with autocast(device_type=cfg.DEVICE.split(':')[0], enabled=(cfg.DEVICE=="cuda"), dtype=cfg.AMP_DTYPE):
                x, y = get_batch(data) # get_batch uses config for device internally
                logits = llm_model(x)
                # Hooks run here automatically during the forward pass, populating current_macro_batch_layer_sparsity
                b, t, c = logits.shape
                logits_view = logits.view(b * t, c)
                y_flat = y.view(b * t)
                loss = criterion(logits_view, y_flat)
                loss = loss / cfg.GRAD_ACCUM_STEPS # Scale loss by GRAD_ACCUM_STEPS from config

            scaler.scale(loss).backward()

        # Clip gradients using grad_clip from config
        if cfg.grad_clip > 0.0:
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(llm_model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update() # Update scaler for next iteration

        # calculate metrics for the *macro* batch
        c_loss = loss.item() * cfg.GRAD_ACCUM_STEPS # Reconstruct approx macro batch loss
        grad_norm = compute_grad_norm(llm_model)

        # Calculate and store average sparsity *per layer* for this iteration
        avg_layer_sparsity = {}
        total_avg_sparsity = 0.0
        num_layers_recorded = 0
        for layer_idx, sparsities in current_macro_batch_layer_sparsity.items():
            if sparsities:
                avg_sparsity_this_layer = sum(sparsities) / len(sparsities)
                avg_layer_sparsity[layer_idx] = avg_sparsity_this_layer
                # Use layer_idx which corresponds to 0..N_LAYERS-1
                all_metrics['layer_sparsity'][layer_idx].append(avg_sparsity_this_layer)
                total_avg_sparsity += avg_sparsity_this_layer
                num_layers_recorded += 1
            else:
                # If a layer had no recorded sparsity (e.g., hook failed), append 0
                # Ensure layer_idx is valid before appending
                if layer_idx < cfg.N_LAYERS:
                    all_metrics['layer_sparsity'][layer_idx].append(0.0)

        # Calculate overall average sparsity across layers for progress bar display
        overall_avg_sparsity = total_avg_sparsity / num_layers_recorded if num_layers_recorded > 0 else 0.0

        # store metrics (except per-layer sparsity, which was stored above)
        all_metrics['loss'].append(c_loss)
        all_metrics['grad_norm'].append(grad_norm)
        all_metrics['lr'].append(lr)

        total_tokens += cfg.TOKENS_PER_BATCH
        iter_num += 1

        if iter_num % cfg.GET_SAMPLE_EVERY == 0 or total_tokens >= cfg.MAX_TOKENS:
            generate_examples(llm_model, iter_num, c_loss, total_tokens, tokenizer)
            save_fig(all_metrics)

        progress_bar.update(cfg.TOKENS_PER_BATCH)
        progress_bar.set_postfix({
            'loss': f"{c_loss:.4f}",
            'sparsity': f"{overall_avg_sparsity:.3f}",
            'norm': f"{grad_norm:.2f}",
            'lr': f"{lr:.1e}",
            'iter': iter_num
        })

except KeyboardInterrupt:
    print("Training interrupted by user.")
finally:
    if hook_handles:
         for handle in hook_handles:
             handle.remove()
         print(f"Removed {len(hook_handles)} sparsity hooks.")

    print("Saving final checkpoint and plots...")

    save_checkpoint(llm_model, all_metrics)

    if iter_num > 0:
        save_layer_sparsity_plot(all_metrics['layer_sparsity'], cfg.N_LAYERS)
        save_fig(all_metrics)
    print("Final saving complete.")
    progress_bar.close()

print("Training finished.")