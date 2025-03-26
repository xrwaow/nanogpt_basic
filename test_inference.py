import torch
from transformers import AutoTokenizer
from model import llm  # Assuming model.py contains the llm class
from config import *   # Import config constants
from load_data import load_checkpoint

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer_name = 'unsloth/Mistral-7B-v0.3'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Model configuration (matching training setup)
VOCAB_SIZE = tokenizer.vocab_size
D_MODEL = 768
N_HEADS = 12
N_EMBD = 3072
N_LAYERS = 12
SEQ_LEN = BLOCK_SIZE  # From config: 1024
USE_BIAS = False

# Example input text
EXAMPLE_TEXTS = [
    "The quick brown fox jumps over",
    "In a world where technology",
    "Once upon a time, there was"
]

def load_model_and_generate(checkpoint_path=f'checkpoints/{SAVE_MODEL_NAME}/{SAVE_MODEL_NAME}.pt'):
    # Load checkpoint
    model = load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint: {total_tokens:,} tokens processed during training")
    #print(f"Final training loss: {metadata['loss'][-1]:.4f}")

    # Set model to evaluation mode
    model.eval()

    # Generate text for each example
    for text in EXAMPLE_TEXTS:
        # Tokenize input
        input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
        
        # Generate continuation
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_tokens=50,         # Generate 50 tokens
                temperature=1,       # Control randomness
                min_p=0.04,            # Minimum probability filter
                echo_back=True         # Include input in output
            )
        
        # Decode to text
        generated_text = tokenizer.decode(generated_ids)
        
        # Print results
        print("\n-----")
        print(f"Input: {text}")
        print(f"Generated: {generated_text}")
        print("-----")

if __name__ == "__main__":
    # Clear GPU memory if using CUDA
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Run inference
    load_model_and_generate("checkpoints/Acheckpoint_999555072/checkpoint_999555072.pt")
