from datasets import load_dataset
import torch
import os
import glob
import time
import math # For ceil in plotting
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming config.py defines BLOCK_SIZE, BATCH_SIZE, DEVICE etc.
from config import * # Make sure config vars are available


def prepare_data_optimized(dataset_name, max_tokens, tokenizer, batch_size=1000, data_dir="data", num_proc=None):
    """Loads, tokenizes, and saves/appends data efficiently up to max_tokens."""
    os.makedirs(data_dir, exist_ok=True)
    if num_proc is None:
        num_proc = max(1, os.cpu_count() // 2) if os.cpu_count() else 1
        # print(f"Using num_proc={num_proc} for dataset processing (if applicable).") # Can be verbose

    existing_files = glob.glob(os.path.join(data_dir, "*.pt"))
    def get_token_count(filepath):
        try:
            return int(os.path.splitext(os.path.basename(filepath))[0])
        except (ValueError, IndexError):
            return -1

    existing_files.sort(key=get_token_count, reverse=True)

    data_tensor = None
    current_tokens = 0
    source_file = None

    for file in existing_files:
        file_tokens = get_token_count(file)
        if file_tokens == -1:
            # print(f"Warning: Skipping invalid filename {file}") # Can be verbose
            continue

        if file_tokens >= max_tokens:
            print(f"Using existing data file {file} with {file_tokens} tokens (>= {max_tokens})")
            try:
                start_load = time.time()
                loaded_data = torch.load(file, map_location='cpu')
                if loaded_data.numel() > max_tokens:
                     # print(f"Trimming loaded data from {loaded_data.numel()} to {max_tokens} tokens.") # Can be verbose
                     data_tensor = loaded_data[:max_tokens].long()
                else:
                     data_tensor = loaded_data.long()
                # print(f"Loaded and finalized in {time.time() - start_load:.2f}s.") # Can be verbose
                return file
            except Exception as e:
                 print(f"Warning: Failed to load or trim {file}: {e}. Trying smaller files...")
                 continue

        elif data_tensor is None:
            print(f"Loading base data from {file} ({file_tokens} tokens)...")
            try:
                start_load = time.time()
                data_tensor = torch.load(file, map_location='cpu')
                if data_tensor.dtype != torch.long:
                    data_tensor = data_tensor.long()
                current_tokens = data_tensor.numel()
                source_file = file
                print(f"Loaded in {time.time() - start_load:.2f}s. Need {max_tokens - current_tokens} more tokens.")
                break
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}. Trying smaller files...")
                data_tensor = None
                current_tokens = 0
                source_file = None

    if data_tensor is None:
        print("No suitable existing data found, starting from scratch.")
        data_tensor = torch.empty((0,), dtype=torch.long, device='cpu')
        current_tokens = 0

    remaining_tokens = max_tokens - current_tokens
    if remaining_tokens <= 0:
        print("Already have enough tokens.")
        if source_file: return source_file
        if data_tensor is not None and data_tensor.numel() >= max_tokens:
             new_filename = os.path.join(data_dir, f"{max_tokens}.pt")
             if not os.path.exists(new_filename) or get_token_count(new_filename) != max_tokens:
                 # print(f"Saving data tensor with exact token count to {new_filename}") # Can be verbose
                 torch.save(data_tensor[:max_tokens], new_filename)
             return new_filename
        else:
             print("Error: No source file and insufficient tokens.")
             return None


    print(f"Need to collect {remaining_tokens} more tokens.")
    print(f"Loading dataset '{dataset_name}'...")

    try:
        dataset = load_dataset(dataset_name, split='train', trust_remote_code=True, streaming=False)
        if 'text' not in dataset.column_names:
             raise ValueError(f"Dataset '{dataset_name}' needs a 'text' column. Found: {dataset.column_names}")
        dataset = dataset.select_columns(['text'])
        total_examples = len(dataset) if hasattr(dataset, '__len__') else None

    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return source_file if source_file else None


    print(f"Tokenizing and appending data (target: {max_tokens} tokens)...")
    eos_token_id = tokenizer.eos_token_id
    add_eos = eos_token_id is not None
    if not add_eos:
         print("Warning: Tokenizer has no eos_token_id. EOS will not be added between documents.")

    new_tokens_list = []
    collected_count = 0
    processed_examples = 0
    start_proc = time.time()
    data_iterator = dataset.iter(batch_size=batch_size)

    pbar_desc = "Tokenizing"
    pbar_total = max_tokens
    pbar_initial = current_tokens
    pbar = tqdm(total=pbar_total, initial=pbar_initial, desc=pbar_desc, unit="tok")

    try:
        for batch in data_iterator:
            texts = [str(t) if t is not None else "" for t in batch["text"]]
            if not texts: continue

            tokenized_outputs = tokenizer(texts)

            batch_token_ids = []
            for ids in tokenized_outputs['input_ids']:
                if add_eos:
                    batch_token_ids.extend(ids + [eos_token_id])
                else:
                    batch_token_ids.extend(ids)

            if not batch_token_ids: continue

            new_tokens_tensor = torch.tensor(batch_token_ids, dtype=torch.long, device='cpu')
            new_tokens_list.append(new_tokens_tensor)
            batch_token_count = new_tokens_tensor.numel()
            collected_count += batch_token_count
            current_total_tokens_estimate = current_tokens + collected_count

            pbar.update(batch_token_count)

            processed_examples += len(texts)

            if current_total_tokens_estimate >= max_tokens:
                # print(f"\nCollected enough tokens ({current_total_tokens_estimate}). Trimming and concatenating...") # Can be verbose
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving collected data...")
    finally:
        pbar.close()

    if new_tokens_list:
        # print("Concatenating new tokens...") # Can be verbose
        start_cat = time.time()

        all_tensors_to_cat = []
        if data_tensor is not None and data_tensor.numel() > 0:
            all_tensors_to_cat.append(data_tensor)
        all_tensors_to_cat.extend(new_tokens_list)

        if all_tensors_to_cat:
             data_tensor = torch.cat(all_tensors_to_cat)
             current_tokens = data_tensor.numel()
             # print(f"Concatenation finished in {time.time() - start_cat:.2f}s. Total tokens before trim: {current_tokens}") # Can be verbose

             if current_tokens > max_tokens:
                  # print(f"Trimming final tensor from {current_tokens} to {max_tokens} tokens.") # Can be verbose
                  data_tensor = data_tensor[:max_tokens]
                  current_tokens = data_tensor.numel()
             elif current_tokens < max_tokens:
                  print(f"\nWarning: Processed {processed_examples} examples, only gathered {current_tokens}/{max_tokens} tokens.")

        else:
            print("Warning: No data tensors to concatenate.")
            data_tensor = None

    else:
        print("No new tokens were added.")
        if data_tensor is None or data_tensor.numel() == 0:
             print("Error: No existing data and no new tokens added.")
             return None


    proc_time = time.time() - start_proc
    # print(f"Tokenization and processing finished in {proc_time:.2f}s.") # Can be verbose

    if data_tensor is None or data_tensor.numel() == 0:
        print("Warning: Final data tensor is empty or None. Nothing to save.")
        if source_file and get_token_count(source_file) == 0:
            try: os.remove(source_file)
            except OSError: pass
        return None

    final_token_count = data_tensor.numel()
    new_filename = os.path.join(data_dir, f"{final_token_count}.pt")

    if os.path.exists(new_filename) and source_file == new_filename and final_token_count == get_token_count(new_filename):
         # print(f"File {new_filename} already exists with the correct token count. No save needed.") # Can be verbose
         return new_filename

    print(f"Saving data to {new_filename} ({final_token_count} tokens)...")
    start_save = time.time()
    try:
        torch.save(data_tensor, new_filename)
        save_time = time.time() - start_save
        # print(f"Data saved in {save_time:.2f}s.") # Can be verbose
    except Exception as e:
        print(f"Error saving file {new_filename}: {e}")
        return source_file if source_file else None

    if source_file and source_file != new_filename:
        print(f"Removing old file {source_file}")
        try:
            os.remove(source_file)
        except OSError as e:
            print(f"Warning: Could not remove old file {source_file}: {e}")

    return new_filename

# --- Batch Loading ---
def get_batch(data, device):
    """Generates a random batch of data (x, y) pairs."""
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Expected data to be a torch.Tensor, but got {type(data)}")
    if data.numel() < BLOCK_SIZE + 1:
        raise ValueError(f"Dataset too small ({data.numel()} tokens) for block size ({BLOCK_SIZE})")

    ix = torch.randint(0, data.numel() - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+1+BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# --- Plotting and Saving Utilities ---

def save_fig(metrics, filename_base, iter_num):
    """Saves a plot of main training metrics (excluding layer sparsity)."""
    base_dir = os.path.join("checkpoints", filename_base)
    os.makedirs(base_dir, exist_ok=True)
    fig_path = os.path.join(base_dir, f"training_metrics_{iter_num}.png")

    # Filter out layer_sparsity for this plot
    metrics_to_plot = {k: v for k, v in metrics.items() if k != 'layer_sparsity'}

    metrics_count = len(metrics_to_plot)
    if metrics_count == 0:
        plt.close()
        return

    plt.figure(figsize=(16, max(5, 3 * metrics_count))) # Adjust size
    cols = 1 # Simple single column layout
    rows = metrics_count

    for i, (metric_name, values) in enumerate(metrics_to_plot.items(), 1):
        if not values: continue

        ax = plt.subplot(rows, cols, i)
        ax.plot(values, label=metric_name)
        ax.set_xlabel('Optimizer Steps (Iterations)')
        formatted_name = metric_name.replace('_', ' ').title()
        ax.set_ylabel(formatted_name)
        ax.set_title(f'{formatted_name} vs. Steps')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        # Optional log scale
        # if metric_name in ['loss', 'grad_norm']: ax.set_yscale('log')

    plt.tight_layout(pad=2.0)
    try:
        plt.savefig(fig_path)
    except Exception as e:
        print(f"Error saving main metrics plot {fig_path}: {e}")
    finally:
        plt.close()

def save_layer_sparsity_plot(layer_sparsity_metrics, n_layers, filename_base, iter_num):
    """Saves a plot showing activation sparsity for each layer's MLP over time."""
    base_dir = os.path.join("checkpoints", filename_base)
    os.makedirs(base_dir, exist_ok=True)
    fig_path = os.path.join(base_dir, f"layer_sparsity_{iter_num}.png")

    if not layer_sparsity_metrics or n_layers == 0:
         print("No layer sparsity data to plot.")
         return

    # Determine grid size for subplots
    cols = 4 # Adjust number of columns as needed
    rows = math.ceil(n_layers / cols)

    plt.figure(figsize=(5 * cols, 4 * rows)) # Adjust figure size based on grid

    plt.suptitle(f'MLP Activation Sparsity per Layer (Iter: {iter_num})', fontsize=16, y=1.02)

    for layer_idx in range(n_layers):
        ax = plt.subplot(rows, cols, layer_idx + 1)
        # Check if data exists for this layer index in the dictionary
        if layer_idx in layer_sparsity_metrics and layer_sparsity_metrics[layer_idx]:
            values = layer_sparsity_metrics[layer_idx]
            ax.plot(values) # No label needed per line, title is sufficient
            ax.set_title(f'Layer {layer_idx}')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Sparsity')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylim(0, 1) # Sparsity is between 0 and 1
        else:
            # Handle cases where a layer might not have data (e.g., hook failed)
            ax.set_title(f'Layer {layer_idx} (No Data)')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make room for suptitle
    try:
        plt.savefig(fig_path)
        # print(f"Saved layer sparsity plot to {fig_path}") # Can be verbose
    except Exception as e:
        print(f"Error saving layer sparsity plot {fig_path}: {e}")
    finally:
        plt.close()


def save_checkpoint(model, metrics, filename_base):
    """Saves model checkpoint, final main metrics plot, final layer sparsity plot, and metrics data."""
    base_dir = os.path.join("checkpoints", filename_base)
    os.makedirs(base_dir, exist_ok=True)

    # --- Save Model State ---
    checkpoint_path = os.path.join(base_dir, f"{filename_base}.pt")
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'model_params': model_to_save.model_params if hasattr(model_to_save, 'model_params') else {},
    }
    try:
        torch.save(checkpoint, checkpoint_path)
        # print(f"Saved model checkpoint to {checkpoint_path}") # Can be verbose
    except Exception as e:
        print(f"Error saving model checkpoint {checkpoint_path}: {e}")

    # --- Save Metrics Data as JSON ---
    metrics_json_path = os.path.join(base_dir, f"{filename_base}_metrics.json")
    metrics_to_save = {}
    for k, vals in metrics.items():
        if k == 'layer_sparsity':
             # Handle the nested dictionary for layer sparsity
             layer_data = {}
             for layer_idx, layer_vals in vals.items():
                 if layer_vals:
                      try:
                          layer_data[layer_idx] = [float(v.item()) if hasattr(v, 'item') else float(v) for v in layer_vals]
                      except Exception as e:
                          print(f"Warning: Could not convert sparsity values for layer {layer_idx}. Skipping. Error: {e}")
             if layer_data:
                 metrics_to_save[k] = layer_data
        elif vals: # Handle other standard metrics (lists)
             try:
                 metrics_to_save[k] = [float(v.item()) if hasattr(v, 'item') else float(v) for v in vals]
             except Exception as e:
                 print(f"Warning: Could not convert metric '{k}' values. Skipping. Error: {e}")

    if metrics_to_save:
        try:
            with open(metrics_json_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=4)
            # print(f"Saved metrics data to {metrics_json_path}") # Can be verbose
        except Exception as e:
            print(f"Error saving metrics JSON {metrics_json_path}: {e}")

    # --- Save Final Main Metrics Plot ---
    final_fig_path = os.path.join(base_dir, "training_metrics_final.png")
    metrics_to_plot = {k: v for k, v in metrics.items() if k != 'layer_sparsity'}
    metrics_count = len(metrics_to_plot)
    if metrics_count > 0:
        plt.figure(figsize=(16, max(5, 3 * metrics_count)))
        cols = 1
        rows = metrics_count
        for i, (metric_name, values) in enumerate(metrics_to_plot.items(), 1):
             if not values: continue
             ax = plt.subplot(rows, cols, i)
             ax.plot(values, label=metric_name)
             ax.set_xlabel('Optimizer Steps (Iterations)')
             formatted_name = metric_name.replace('_', ' ').title()
             ax.set_ylabel(formatted_name)
             ax.set_title(f'Final {formatted_name} vs. Steps')
             ax.grid(True, linestyle='--', alpha=0.6)
             ax.legend()
             # if metric_name in ['loss', 'grad_norm']: ax.set_yscale('log')
        plt.tight_layout(pad=2.0)
        try:
            plt.savefig(final_fig_path)
            # print(f"Saved final main metrics plot to {final_fig_path}") # Can be verbose
        except Exception as e:
            print(f"Error saving final main metrics plot {final_fig_path}: {e}")
        finally:
            plt.close()

    # --- Save Final Layer Sparsity Plot ---
    # Need N_LAYERS - get it from the keys of layer_sparsity if available, or model_params
    n_layers = len(metrics.get('layer_sparsity', {}).keys())
    if n_layers == 0 and 'model_params' in checkpoint and 'n_layers' in checkpoint['model_params']:
         n_layers = checkpoint['model_params']['n_layers']

    if n_layers > 0 and 'layer_sparsity' in metrics:
         final_sparsity_fig_path = os.path.join(base_dir, "layer_sparsity_final.png")
         save_layer_sparsity_plot(metrics['layer_sparsity'], n_layers, filename_base, "final") # Use "final" in filename


def load_checkpoint(filename_path):
    """Loads a model checkpoint."""
    if not os.path.exists(filename_path):
        raise FileNotFoundError(f"Checkpoint file not found: {filename_path}")

    print(f"Loading checkpoint from: {filename_path}")
    checkpoint = torch.load(filename_path, map_location='cpu')

    if 'model_params' not in checkpoint:
         raise KeyError("Checkpoint missing 'model_params'. Cannot reconstruct model.")
    params = checkpoint['model_params']

    # Import llm class locally if needed, or ensure it's globally available
    # from model import llm # Assuming model.py defines llm
    model = llm(**params, yap=False) # Create model without printing params

    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    state_dict = checkpoint['model_state_dict']

    unwrapped_state_dict = {}
    prefix = '_orig_mod.'
    prefix_len = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            unwrapped_state_dict[k[prefix_len:]] = v
        else:
            unwrapped_state_dict[k] = v

    try:
         model.load_state_dict(unwrapped_state_dict, strict=True)
    except RuntimeError as e:
         print(f"Warning: Error loading state_dict (strict=True): {e}")
         print("Attempting to load with strict=False...")
         try:
             model.load_state_dict(unwrapped_state_dict, strict=False)
             print("Loaded state_dict with strict=False.")
         except Exception as e2:
              print(f"Failed to load state_dict even with strict=False: {e2}")
              raise e # Reraise original error if strict=False also fails badly


    model.to(DEVICE)
    print(f"Model loaded successfully to {DEVICE}.")
    return model

def generate_examples(model, checkpoint_base_name, iter_num, loss, total_tokens, tokenizer):
    """Generates text examples using the model and saves them to a file."""
    prompts = [
        "The capital city of Japan is",
        "The quick brown fox jumps over the",
        "In a world where technology advances rapidly,",
        "Artificial intelligence is",
    ]
    gen_params = { "max_tokens": 48, "temperature": 0.7, "top_k": 50 }
    param_variations = [
         {"temperature": 0.5, "top_k": 40},
         {"temperature": 0.8, "top_k": 100},
         {"temperature": 1.0, "top_k": None}, # More random
    ]

    output_str = f"===== Generation at Iter: {iter_num}, Loss: {loss:.4f}, Total Tokens: {total_tokens:,} =====\n\n"

    for prompt_text in prompts:
        output_str += f"--- Prompt: \"{prompt_text}\" ---\n"
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt').to(DEVICE)

        for i, params in enumerate(param_variations):
            combined_params = {**gen_params, **params}
            # Ensure model is in eval mode and no_grad is active (handled by model.generate)
            generated_ids = model.generate(prompt_tokens, **combined_params, echo_back=False)
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            param_desc = ", ".join([f"{k}={v}" for k,v in params.items()])
            output_str += f"Params: ({param_desc})\n"
            output_str += f"Output: |{generated_text.strip()}\n\n"
        output_str += "---\n\n"

    checkpoint_dir = os.path.join("checkpoints", checkpoint_base_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    file_path = os.path.join(checkpoint_dir, "example_generations.txt")

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(output_str)
    except Exception as e:
        print(f"Error writing generation examples to {file_path}: {e}")