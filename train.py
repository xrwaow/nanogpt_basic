import torch
from transformers import AutoTokenizer
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from config import *
from load_data import *
from model import llm

def calculate_entropy(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy.mean().item()

def compute_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return torch.sqrt(torch.tensor(total_norm)).item()

def main():
    tokenizer_name = 'unsloth/mistral-7b-v0.3'
    dataset_name = '/home/xr/.cache/huggingface/datasets/haritzpuerto___the_pile_00_open_web_text2/' # 'haritzpuerto/the_pile_00_openwebtext2'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data_path = prepare_data(dataset_name, MAX_TOKENS, tokenizer)
    data = torch.load(data_path, weights_only=True)

    # create model
    llm_model = llm(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_embd=N_EMBD,
        n_layers=N_LAYERS,
        seq_len=BLOCK_SIZE,
        use_bias=USE_BIAS
    ).to(DEVICE)

    llm_model = torch.compile(llm_model)

    optimizer = torch.optim.AdamW(
        llm_model.parameters(), 
        lr=1e-3, 
        fused=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")
    
    # initialize tracking
    total_tokens = 0
    progress_bar = tqdm(total=MAX_TOKENS, desc='training progress')
    all_metrics = {'loss': [], 'entropy': [], 'perplexity': [], 'grad_norm': []}

    llm_model.train()
    
    # training loop
    while total_tokens < MAX_TOKENS:
        optimizer.zero_grad()
        
        for _ in range(GRAD_ACCUM_STEPS):
            with autocast("cuda"):
                x, y = get_batch(data, DEVICE)
                logits = llm_model(x)
                b, t, c = logits.shape
                logits = logits.view(b * t, c)
                y_flat = y.view(b * t)
                loss = criterion(logits, y_flat) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        
        # calculate metrics
        c_loss = loss.item() * GRAD_ACCUM_STEPS
        entropy = calculate_entropy(logits.detach())
        perplexity = torch.exp(torch.tensor(c_loss)).item()
        grad_norm = compute_grad_norm(llm_model)
        
        # store metrics
        all_metrics['loss'].append(c_loss)
        all_metrics['entropy'].append(entropy)
        all_metrics['perplexity'].append(perplexity)
        all_metrics['grad_norm'].append(grad_norm)
        
        total_tokens += TOKENS_PER_BATCH
        progress_bar.update(TOKENS_PER_BATCH)
        progress_bar.set_postfix({'loss': c_loss, 'entropy': entropy, 'perplexity': perplexity})
    
    save_checkpoint(llm_model, all_metrics, SAVE_MODEL_NAME)

if __name__ == "__main__":
    main()