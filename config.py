D_MODEL=768
N_HEADS=12
N_EMBD=3072
N_LAYERS=12

total_tokens = 1e7 # 1e8

BLOCK_SIZE =    1024
BATCH_SIZE =      24   # 16
GRAD_ACCUM_STEPS = 4   # 8

USE_BIAS = False
DEVICE = "cuda"

TOKENS_PER_BATCH = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS
MAX_TOKENS = int((total_tokens // (TOKENS_PER_BATCH)) * (TOKENS_PER_BATCH))

SAVE_MODEL_NAME = f"checkpoint_{MAX_TOKENS}"

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