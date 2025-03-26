from datasets import load_dataset
import torch
import os
from tqdm import tqdm
from config import *
import json
import matplotlib.pyplot as plt
import glob

def save_checkpoint(model, metrics, filename):
    base_dir = f"checkpoints/{filename}"
    
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_params': model.model_params,
    }
    with open(f'{base_dir}/{filename}.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)

    plt.figure(figsize=(15, 10))
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
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

def get_batch(data, device):
    ix = torch.randint(0, len(data) - BLOCK_SIZE + 1, (BATCH_SIZE,), device=device)
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+1+BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)

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