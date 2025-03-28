from datasets import load_dataset
import torch
import os
from tqdm import tqdm
from config import *
import json
import matplotlib.pyplot as plt
import glob
import time
import math

def prepare_data_optimized(dataset_name, max_tokens, tokenizer, batch_size=1000, data_dir="data", num_proc=None):
    os.makedirs(data_dir, exist_ok=True)
    if num_proc is None:
        num_proc = os.cpu_count()

    existing_files = glob.glob(os.path.join(data_dir, "*.pt"))
    # Use more robust parsing for filename (handles potential dots in dir names)
    def get_token_count(filepath):
        try:
            return int(os.path.splitext(os.path.basename(filepath))[0])
        except ValueError:
            return -1 # Invalid filename format

    existing_files.sort(key=get_token_count)

    data_tensor = None
    current_tokens = 0
    source_file = None

    # Check existing files (largest first)
    for file in reversed(existing_files):
        file_tokens = get_token_count(file)
        if file_tokens == -1:
            print(f"Warning: Skipping invalid filename {file}")
            continue

        if file_tokens >= max_tokens:
            print(f"Using existing data file {file} with {file_tokens} tokens (>= {max_tokens})")
            # Optional: Trim if strictly needed, but often okay to have slightly more
            # Consider loading and trimming if exact count is critical downstream
            return file
        elif data_tensor is None: # Found the largest file < max_tokens
            print(f"Loading base data from {file} ({file_tokens} tokens)...")
            try:
                start_load = time.time()
                # Load directly as a tensor, ensuring it's on CPU
                data_tensor = torch.load(file, map_location='cpu')
                # Ensure it's the right type if loaded from older formats potentially
                if data_tensor.dtype != torch.long:
                    data_tensor = data_tensor.long()
                current_tokens = data_tensor.numel()
                source_file = file
                print(f"Loaded in {time.time() - start_load:.2f}s. Need {max_tokens - current_tokens} more tokens.")
                # Keep searching in case of corrupted file, but this is our best candidate so far
                # If loading fails, the loop continues to smaller files.
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}. Trying smaller files...")
                # Reset state as this file is unusable
                data_tensor = None
                current_tokens = 0
                source_file = None

    if data_tensor is None:
        print("No suitable existing data found, starting from scratch.")
        # Initialize as an empty tensor of the correct type
        data_tensor = torch.empty((0,), dtype=torch.long)
        current_tokens = 0

    remaining_tokens = max_tokens - current_tokens
    if remaining_tokens <= 0:
        print("Already have enough tokens.")
        return source_file # Should be the largest file loaded
    print(f"Loading dataset '{dataset_name}'...")
    
    try:
        dataset = load_dataset(dataset_name, split='train', trust_remote_code=True) # Add trust_remote_code for some datasets
        if 'text' not in dataset.column_names:
             raise ValueError(f"Dataset '{dataset_name}' does not contain a 'text' column. Available: {dataset.column_names}")
        dataset = dataset.select_columns(['text']) # Keep only text

    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return source_file # Return whatever we had before failing


    print(f"Tokenizing and appending data (target: {max_tokens} tokens)...")
    eos_token_id = tokenizer.eos_token_id
    add_eos = eos_token_id is not None
    if not add_eos:
         print("Warning: Tokenizer does not have a defined eos_token_id. EOS will not be added.")


    # --- Batch Tokenization Logic ---
    new_tokens_list = []
    collected_count = 0
    processed_examples = 0
    total_examples = len(dataset)
    start_proc = time.time()

    # Iterate through the dataset in batches
    data_iterator = dataset.iter(batch_size=batch_size)
    
    # Use tqdm on the iterator if possible, estimate total batches
    total_batches = math.ceil(total_examples / batch_size)
    pbar = tqdm(total=max_tokens, initial=current_tokens, desc="Tokenizing", unit="tok")

    try:
        for batch in data_iterator:
            texts = [str(t) if t is not None else "" for t in batch["text"]]
            if not texts: continue # Skip empty batches

            # Batch tokenize
            tokenized_outputs = tokenizer(texts) # Returns dict {'input_ids': [[...], [...]], 'attention_mask': ...}

            batch_token_ids = []
            for ids in tokenized_outputs['input_ids']:
                if add_eos:
                    batch_token_ids.extend(ids + [eos_token_id])
                else:
                    batch_token_ids.extend(ids)

            new_tokens_list.append(torch.tensor(batch_token_ids, dtype=torch.long))
            collected_count += len(batch_token_ids)
            current_total_tokens = current_tokens + collected_count

            pbar.update(len(batch_token_ids)) # Update progress bar by tokens added in this batch

            processed_examples += len(texts)

            # Check if we have enough tokens
            if current_total_tokens >= max_tokens:
                print(f"\nCollected enough tokens ({current_total_tokens}). Trimming and concatenating...")
                break # Exit the loop

            # Optional: Periodically concatenate to manage memory if intermediate list gets huge
            # e.g., if collected_count > 50_000_000: concatenate and clear list

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving collected data...")
        # Proceed to concatenation/saving with what we have
    finally:
        pbar.close()


    # Concatenate all collected new tokens
    if new_tokens_list:
        print("Concatenating new tokens...")
        start_cat = time.time()
        # new_data_tensor = torch.cat(new_tokens_list) # Causes intermediate memory spike
        # More memory efficient cat (approximate):
        temp_tensors = []
        current_size = current_tokens
        needed = max_tokens - current_tokens
        
        if needed <= 0 and data_tensor is not None: # Should not happen if loop logic is correct
             pass # Already had enough
        else:
             # Add existing tensor first if it exists
             if data_tensor is not None and data_tensor.numel() > 0:
                 temp_tensors.append(data_tensor)

             for t in new_tokens_list:
                 if needed <= 0: break
                 if t.numel() <= needed:
                     temp_tensors.append(t)
                     needed -= t.numel()
                     current_size += t.numel()
                 else: # This tensor contains the cutoff point
                     temp_tensors.append(t[:needed])
                     current_size += needed
                     needed = 0
                     break
            
             # Final concatenation
             if temp_tensors:
                 data_tensor = torch.cat(temp_tensors)
                 current_tokens = data_tensor.numel()
             # Ensure exact size if we overshot slightly due to batching
             if current_tokens > max_tokens:
                  data_tensor = data_tensor[:max_tokens]
                  current_tokens = data_tensor.numel()
             elif needed > 0 : # We processed everything but didn't reach max_tokens
                 print(f"\nWarning: Processed {processed_examples}/{total_examples} examples, but only gathered {current_tokens}/{max_tokens} tokens.")


        print(f"Concatenation finished in {time.time() - start_cat:.2f}s. Final token count: {current_tokens}")

    else: # No new tokens were added
        print("No new tokens were added.")
        if source_file: return source_file # Return the original file
        else:
            print("Error: No data loaded and no new tokens added.")
            return None # Indicate failure


    proc_time = time.time() - start_proc
    print(f"Tokenization and processing finished in {proc_time:.2f}s.")

    # --- Saving Data ---
    # Ensure final tensor exists and has content before saving
    if data_tensor is None or data_tensor.numel() == 0:
        print("Warning: Final data tensor is empty. Nothing to save.")
        # Clean up potentially empty source file reference if we started from scratch
        if source_file and get_token_count(source_file) == 0:
            try: os.remove(source_file)
            except OSError: pass
        return None

    new_filename = os.path.join(data_dir, f"{current_tokens}.pt")

    # Avoid saving if the file already exists with the exact same token count
    if os.path.exists(new_filename) and source_file == new_filename:
         print(f"File {new_filename} already exists with the correct token count. No changes needed.")
         return new_filename

    print(f"Saving data to {new_filename} ({current_tokens} tokens)...")
    start_save = time.time()
    try:
        torch.save(data_tensor, new_filename)
        save_time = time.time() - start_save
        print(f"Data saved in {save_time:.2f}s.")
    except Exception as e:
        print(f"Error saving file {new_filename}: {e}")
        # Attempt to return the original file if saving failed
        return source_file if source_file else None


    # Remove old file if we appended and created a new file name
    if source_file and source_file != new_filename:
        print(f"Removing old file {source_file}")
        try:
            os.remove(source_file)
        except OSError as e:
            print(f"Warning: Could not remove old file {source_file}: {e}")

    return new_filename

def prepare_data(dataset_name, max_tokens, tokenizer):
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Find all existing data files
    existing_files = glob.glob("data/*.pt")
    existing_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Check if we have a suitable existing file
    for file in existing_files:
        file_tokens = int(os.path.basename(file).split('.')[0])
        if file_tokens >= max_tokens:
            print(f"Using existing data file {file} with {file_tokens} tokens")
            return file
    
    # If we get here, we need to create/append to a file
    if existing_files:
        # Use the largest existing file as starting point
        largest_file = existing_files[-1]
        current_tokens = int(os.path.basename(largest_file).split('.')[0])
        data = torch.load(largest_file).tolist()
        print(f"Found existing data with {current_tokens} tokens, appending...")
    else:
        current_tokens = 0
        data = []
        print("No existing data found, starting from scratch")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split='train')
    
    # Calculate how many more tokens we need
    remaining_tokens = max_tokens - current_tokens
    if remaining_tokens <= 0:
        return largest_file
    
    # Process dataset to get more tokens
    for example in tqdm(dataset, desc="Processing data"):
        new_tokens = tokenizer.encode(example["text"]) + [tokenizer.eos_token_id]
        data.extend(new_tokens)
        current_tokens += len(new_tokens)
        
        if current_tokens >= max_tokens:
            data = data[:max_tokens]
            current_tokens = max_tokens
            break
    
    # Save the new data file
    new_filename = f"data/{current_tokens}.pt"
    torch.save(torch.tensor(data), new_filename)
    
    # Remove old file if we appended to it (keep only the largest file)
    if existing_files and largest_file != new_filename:
        os.remove(largest_file)
    
    print(f"Data saved to {new_filename} with {current_tokens} tokens")
    return new_filename

def get_batch(data, device):
    ix = torch.randint(0, len(data) - BLOCK_SIZE + 1, (BATCH_SIZE,), device=device)
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+1+BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)

def save_fig(metrics, filename, iter_num):
    base_dir = f"checkpoints/{filename}"
    plt.figure(figsize=(16, 9))
    metrics_count = len(metrics)

    rows = (metrics_count + 1) // 2  # Ceiling division to get enough rows
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(rows, 2, i)
        plt.plot(values, label=metric_name)
        plt.xlabel('batch updates')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over time')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{base_dir}/training_metrics_{iter_num}.png')
    plt.close()

def save_checkpoint(model, metrics, filename):
    base_dir = f"checkpoints/{filename}"
    
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_params': model.model_params,
    }
    with open(f'{base_dir}/{filename}.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)

    #plt.figure(figsize=(15, 10))
    plt.figure(figsize=(16, 9))
    metrics_count = len(metrics)

    rows = (metrics_count + 1) // 2  # Ceiling division to get enough rows
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(rows, 2, i)
        plt.plot(values, label=metric_name)
        plt.xlabel('batch updates')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over time')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{base_dir}/training_metrics.png')
    plt.close()

    torch.save(checkpoint, f'{base_dir}/{filename}.pt')

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    
    # get model params and create new model instance
    params = checkpoint['model_params']
    model = llm(**params).to(DEVICE)
    
    # fix the state dict keys
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('_orig_mod.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    return model

def generate_examples(model, checkpoint_path, idx, loss, total_tokens, tokenizer):
    examples = [
        "The capital city of Japan is",
        "The quick brown fox jumps over",
        "In a world where technology"
    ]

    params = {"temperature": [0.5, 1, 1], "min_p": [0.02, 0.02, 0.08]}
    ret = f"generation at {idx}, loss: {loss:2.4f}, t_tokens: {total_tokens}\n-----\n"
    
    for example in examples:
        example_tokenized = tokenizer.encode(example, return_tensors='pt').to(DEVICE)
        for i in range(3):
            out = model.generate(example_tokenized, max_tokens=32, temperature=params['temperature'][i], min_p=params['min_p'][i], echo_back=False)
            ret += f"temp: {params['temperature'][i]:1.1f} min_p: {params['min_p'][i]:1.2f}\n{example}|{tokenizer.decode(out)}\n"
        ret += "---\n"
    ret += "\n"
    
    checkpoint_dir = f"checkpoints/{checkpoint_path}"
    file_path = f"{checkpoint_dir}/example_output.txt"

    # create dir if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # create file if it doesn't exist
    if not os.path.exists(file_path):
        open(file_path, 'w').close()

    # now write to file
    with open(file_path, "a") as f:
        f.write(ret)
