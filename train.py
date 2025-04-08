import torch
from transformers import AutoTokenizer
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import math
import os
import functools # Added for hook factory

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

import sys
if sys.argv[1] == "a100":
    from config_a100 import *
else:
    from config import *
from load_data import * # Imports save_fig, save_checkpoint, and the new save_layer_sparsity_plot
from model import llm

torch.manual_seed(SEED)

# Global dict to store activation sparsity values per layer for the current macro batch
# Keys: layer_idx, Values: list of sparsity values from micro-batches
current_macro_batch_layer_sparsity = {}

def sparsity_hook_factory(layer_idx):
    """Creates a hook function that knows its layer index."""
    def sparsity_hook(module, input, output):
        """Forward hook to calculate activation sparsity for a specific layer."""
        threshold = 1e-3 # Define the threshold for 'zero' activation
        is_zero = torch.abs(output) < threshold
        sparsity = torch.mean(is_zero.float()).item()
        # Append to the list for the specific layer within the current macro batch
        if layer_idx not in current_macro_batch_layer_sparsity:
             current_macro_batch_layer_sparsity[layer_idx] = []
        current_macro_batch_layer_sparsity[layer_idx].append(sparsity)
    return sparsity_hook

# Removed calculate_entropy function

def compute_grad_norm(model):
    total_norm = 0.0 # Use float for accumulation
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5 # Calculate sqrt at the end
    return total_norm

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

data_path = prepare_data_optimized(dataset_name, MAX_TOKENS, tokenizer)
data = torch.load(data_path, map_location='cpu', weights_only=False)
if data.device != torch.device('cpu'):
    data = data.to('cpu')
if data.dtype != torch.long:
    data = data.long()

# create model
llm_model = llm(
    vocab_size=tokenizer.vocab_size,
    n_heads=N_HEADS,
    n_embd=N_EMBD,
    n_layers=N_LAYERS,
    seq_len=BLOCK_SIZE,
    use_bias=USE_BIAS,
    kv_heads=6,
)

# --- Register Per-Layer Sparsity Hooks BEFORE moving to device and compiling ---
hook_handles = []
# Ensure model has 'blocks' attribute and it's iterable
if hasattr(llm_model, 'blocks') and isinstance(llm_model.blocks, torch.nn.ModuleList):
    for i, block in enumerate(llm_model.blocks):
        try:
            # Target the GELU activation in each MLP block's sequential net (index 1)
            target_module = block.mlp.net[1]
            hook_func = sparsity_hook_factory(i)
            handle = target_module.register_forward_hook(hook_func)
            hook_handles.append(handle)
            print(f"Registered sparsity hook on layer {i} MLP GELU.")
        except AttributeError:
             print(f"Warning: Could not find mlp.net[1] in block {i}. Skipping hook.")
        except Exception as e:
            print(f"Warning: Could not register sparsity hook for layer {i}: {e}")
else:
    print("Warning: Model does not have 'blocks' attribute or it's not a ModuleList. No sparsity hooks registered.")
# --- End Hook Registration ---

llm_model = llm_model.to(DEVICE) # Move to device *before* compile
llm_model = torch.compile(llm_model) # Compile the model

optimizer = torch.optim.AdamW(
    llm_model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
    fused=(DEVICE=="cuda") # Fused might only work on CUDA
)

criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler(enabled=(DEVICE == "cuda")) # Explicitly enable only for CUDA

# initialize tracking
total_tokens = 0
# Removed 'entropy', added 'layer_sparsity' dictionary
all_metrics = {
    'loss': [],
    'grad_norm': [],
    'lr': [],
    'layer_sparsity': {layer_idx: [] for layer_idx in range(N_LAYERS)} # Store list for each layer
}
progress_bar = tqdm(total=MAX_TOKENS, desc='training progress', unit='tok')
iter_num = 0

# training loop
try: # Wrap training loop for cleanup
    while total_tokens < MAX_TOKENS:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Reset per-layer sparsity list for the *macro* batch
        current_macro_batch_layer_sparsity.clear()

        optimizer.zero_grad(set_to_none=True) # More memory efficient

        for micro_step in range(GRAD_ACCUM_STEPS):
            with autocast(device_type=DEVICE.split(':')[0], enabled=(DEVICE=="cuda"), dtype=torch.bfloat16):
                x, y = get_batch(data, DEVICE)
                logits = llm_model(x)
                # Hooks run here automatically during the forward pass, populating current_macro_batch_layer_sparsity
                b, t, c = logits.shape
                logits_view = logits.view(b * t, c)
                y_flat = y.view(b * t)
                loss = criterion(logits_view, y_flat)
                loss = loss / GRAD_ACCUM_STEPS # Scale loss

            scaler.scale(loss).backward()

        if grad_clip > 0.0:
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(llm_model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update() # Update scaler for next iteration

        # calculate metrics for the *macro* batch
        c_loss = loss.item() * GRAD_ACCUM_STEPS # Reconstruct approx macro batch loss
        grad_norm = compute_grad_norm(llm_model)

        # Calculate and store average sparsity *per layer* for this iteration
        avg_layer_sparsity = {}
        total_avg_sparsity = 0.0
        num_layers_recorded = 0
        for layer_idx, sparsities in current_macro_batch_layer_sparsity.items():
            if sparsities:
                avg_sparsity_this_layer = sum(sparsities) / len(sparsities)
                avg_layer_sparsity[layer_idx] = avg_sparsity_this_layer
                all_metrics['layer_sparsity'][layer_idx].append(avg_sparsity_this_layer)
                total_avg_sparsity += avg_sparsity_this_layer
                num_layers_recorded += 1
            else:
                 # If a layer had no recorded sparsity (e.g., hook failed), append 0
                 all_metrics['layer_sparsity'][layer_idx].append(0.0)

        # Calculate overall average sparsity across layers for progress bar display
        overall_avg_sparsity = total_avg_sparsity / num_layers_recorded if num_layers_recorded > 0 else 0.0

        # store metrics (except per-layer sparsity, which was stored above)
        all_metrics['loss'].append(c_loss)
        all_metrics['grad_norm'].append(grad_norm)
        all_metrics['lr'].append(lr)

        total_tokens += TOKENS_PER_BATCH
        iter_num += 1

        # Log and save checkpoints/samples periodically
        if iter_num % GET_SAMPLE_EVERY == 0 or total_tokens >= MAX_TOKENS:
            llm_model.eval()
            generate_examples(llm_model, SAVE_MODEL_NAME, iter_num, c_loss, total_tokens, tokenizer)
            llm_model.train()
            # Save the main metrics plot
            save_fig(all_metrics, SAVE_MODEL_NAME, iter_num)
            # Save the new per-layer sparsity plot
            #save_layer_sparsity_plot(all_metrics['layer_sparsity'], N_LAYERS, SAVE_MODEL_NAME, iter_num)

        if iter_num % SAVE_EVERY == 0 or total_tokens >= MAX_TOKENS:
            save_checkpoint(llm_model, all_metrics, SAVE_MODEL_NAME) # save_checkpoint also saves final plots

        progress_bar.update(TOKENS_PER_BATCH)
        # Removed entropy, added overall avg sparsity to postfix
        progress_bar.set_postfix({
            'loss': f"{c_loss:.4f}",
            'sparsity': f"{overall_avg_sparsity:.3f}", # Show overall average sparsity
            'norm': f"{grad_norm:.2f}",
            'lr': f"{lr:.1e}",
            'iter': iter_num
        })

except KeyboardInterrupt:
    print("Training interrupted by user.")
finally:
    # --- Clean up the hooks ---
    if hook_handles:
         for handle in hook_handles:
             handle.remove()
         print(f"Removed {len(hook_handles)} sparsity hooks.")
    # --- End Hook Cleanup ---

    # Save final state upon completion or interruption
    print("Saving final checkpoint and plots...")
    save_checkpoint(llm_model, all_metrics, SAVE_MODEL_NAME) # Saves model and final plots
    # Save final layer sparsity plot explicitly if save_checkpoint doesn't handle it
    if iter_num > 0 : # Only save if training ran at least one iteration
         save_layer_sparsity_plot(all_metrics['layer_sparsity'], N_LAYERS, SAVE_MODEL_NAME, iter_num)

    print("Final saving complete.")
    progress_bar.close()

print("Training finished.")