N_LAYERS=   12
N_HEADS =   12
N_EMBD  =  768

total_tokens =   1e9

BLOCK_SIZE =    1024
BATCH_SIZE =      92
GRAD_ACCUM_STEPS = 4

TOKENS_PER_BATCH = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS
MAX_TOKENS = int((total_tokens // (TOKENS_PER_BATCH)) * (TOKENS_PER_BATCH))

GET_SAMPLE_EVERY = 50
SAVE_EVERY = GET_SAMPLE_EVERY * 4

SEED = 42

# adamw optimizer
learning_rate = 6e-4 # max learning rate
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 10#2000 # how many steps to warm up for
lr_decay_iters = MAX_TOKENS // TOKENS_PER_BATCH # 600000 # should be ~= max_iters per Chinchilla

# dropout = 0
USE_BIAS = False
DEVICE = "cuda"

SAVE_MODEL_NAME = f"checkpoint_fineweb_1B_tied_{MAX_TOKENS}"

tokenizer_name = 'unsloth/mistral-7b-v0.3'
dataset_name = 'tensorlabco/fineweb-edu-sample-10BT'

"""
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L207
config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
"""