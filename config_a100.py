import torch

# --- Model Architecture ---
N_LAYERS=   12
N_HEADS =   12
KV_HEADS =   2
N_EMBD  =  768
USE_BIAS = False
ROPE_THETA = 10000.0 # RoPE base frequency
TIE_WEIGHTS = True # Tie embedding and output projection weights

# --- Training Hyperparameters ---
total_tokens =   1e9
BLOCK_SIZE =    1024
BATCH_SIZE =      92
GRAD_ACCUM_STEPS = 4

# Derived training parameters
TOKENS_PER_BATCH = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS
MAX_TOKENS = int((total_tokens // (TOKENS_PER_BATCH)) * (TOKENS_PER_BATCH))

# --- Optimizer (AdamW) ---
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 1e-2#1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# --- Learning Rate Schedule ---
decay_lr = True
warmup_iters = 10
lr_decay_iters = MAX_TOKENS // TOKENS_PER_BATCH # Steps for cosine decay (typically max_iters)

# --- Checkpointing & Logging ---
GET_SAMPLE_EVERY = 50
SAVE_EVERY = GET_SAMPLE_EVERY * 4
CHECKPOINT_DIR = "checkpoints"

SAVE_MODEL_NAME = f"testing_fineweb_{MAX_TOKENS // 1_000_000}M"#f"llm_L{N_LAYERS}_H{N_HEADS}_KV{KV_HEADS}_E{N_EMBD}_T{MAX_TOKENS // 1_000_000}M"

# --- Data ---
DATA_DIR = "data"
dataset_name = 'tensorlabco/fineweb-edu-sample-10BT'
tokenizer_name = 'unsloth/mistral-7b-v0.3'
DATA_PREP_BATCH_SIZE = 1000 # Batch size for data loading/tokenization
DATA_PREP_NUM_PROC = None # Num processes for dataset mapping (None for auto)

# --- Environment & Reproducibility ---
SEED = 42
DEVICE = "cuda"
AMP_DTYPE = torch.bfloat16

# --- Misc ---
PRINT_MODEL_PARAMS = True#False

# --- Assertions for Sanity ---
assert N_HEADS % KV_HEADS == 0, "N_HEADS must be divisible by KV_HEADS"
assert lr_decay_iters >= warmup_iters, "lr_decay_iters must be >= warmup_iters"