import torch
from transformers import AutoTokenizer
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import math

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from config import *
from load_data import *
from model import llm

def calculate_entropy(logits):
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]  # Stabilize
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-8, max=1.0)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy.mean().item()

def compute_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return torch.sqrt(torch.tensor(total_norm)).item()

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

data_path = prepare_data_optimized(dataset_name, MAX_TOKENS, tokenizer)#prepare_data(dataset_name, MAX_TOKENS, tokenizer)
data = torch.load(data_path, weights_only=True)

# create model
llm_model = llm(
    vocab_size=tokenizer.vocab_size,
    n_heads=N_HEADS,
    n_embd=N_EMBD,
    n_layers=N_LAYERS,
    seq_len=BLOCK_SIZE,
    use_bias=USE_BIAS
).to(DEVICE)

llm_model = torch.compile(llm_model)

optimizer = torch.optim.AdamW(
    llm_model.parameters(), 
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
    fused=True
)

criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler(enabled=True)

# initialize tracking
total_tokens = 0
#all_metrics = {'loss': [], 'entropy': [], 'perplexity': [], 'grad_norm': [], 'lr': []}
all_metrics = {'loss': [], 'entropy': [], 'grad_norm': [], 'lr': []}
progress_bar = tqdm(total=MAX_TOKENS, desc='training progress')
iter_num = 0

# training loop
while total_tokens < MAX_TOKENS:
    # determine and set the learning rate for this iteration
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = learning_rate
    
    optimizer.zero_grad()
    
    for _ in range(GRAD_ACCUM_STEPS):
        with autocast("cuda", enabled=True, dtype=torch.bfloat16):
            x, y = get_batch(data, DEVICE)
            logits = llm_model(x)
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            y_flat = y.view(b * t)
            loss = criterion(logits, y_flat) / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
    
    # clip gradients if grad_clip is set
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(llm_model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    
    # calculate metrics
    c_loss = loss.item() * GRAD_ACCUM_STEPS
    entropy = calculate_entropy(logits.detach())
    #perplexity = torch.exp(torch.tensor(c_loss)).item()
    grad_norm = compute_grad_norm(llm_model)
    
    # store metrics
    # add Throughput (Samples/sec or Tokens/sec)
    all_metrics['loss'].append(c_loss)
    all_metrics['entropy'].append(entropy)
    #all_metrics['perplexity'].append(perplexity)
    all_metrics['grad_norm'].append(grad_norm)
    all_metrics['lr'].append(lr)
    
    total_tokens += TOKENS_PER_BATCH
    iter_num += 1

    if iter_num % GET_SAMPLE_EVERY == 0 or total_tokens >= MAX_TOKENS:
        generate_examples(llm_model, SAVE_MODEL_NAME, iter_num, c_loss, total_tokens, tokenizer)
        save_fig(all_metrics, SAVE_MODEL_NAME, iter_num)
    
    if iter_num % SAVE_EVERY == 0 or total_tokens >= MAX_TOKENS:
        save_checkpoint(llm_model, all_metrics, SAVE_MODEL_NAME)
    
    progress_bar.update(TOKENS_PER_BATCH)
    progress_bar.set_postfix({
        'loss': c_loss, 
        'entropy': entropy, 
        #'perplexity': perplexity,
        'lr': lr,
        'iter': iter_num
    })

save_checkpoint(llm_model, all_metrics, SAVE_MODEL_NAME)