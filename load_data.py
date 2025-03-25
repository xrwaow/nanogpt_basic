from datasets import load_dataset
import torch
import os
from tqdm import tqdm
from config import *
from model import llm
import json
import matplotlib.pyplot as plt

def save_checkpoint(model, metrics, filename):
    base_dir = f"checkpoints/{filename}"
    
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_params': model.model_params,
        #'metrics': metrics
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
    return model#, checkpoint.get('metrics', None)

def get_batch(data, device):
    ix = torch.randint(0, len(data) - BLOCK_SIZE + 1, (BATCH_SIZE,), device=device) # would device=device help
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+1+BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)

def prepare_data(dataset_name, max_tokens, tokenizer):
    data_path = f"data/{max_tokens}.pt"
    
    if os.path.exists(data_path):
        if int(data_path.split("/")[1][:-3]) == max_tokens:
            print(f"{data_path} already exists")
            return data_path
    
    data = load_dataset(dataset_name, split='train')
    ret = []
    for example in tqdm(data):
        ret += tokenizer.encode(example["text"]) + [tokenizer.eos_token_id]
        if len(ret) >= max_tokens: break
    
    tensor_data = torch.tensor(ret)
    torch.save(tensor_data, data_path)
    print("data saved")
    return data_path