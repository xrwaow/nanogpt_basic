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

# Example input text
EXAMPLE_TEXTS = [
    "The quick brown fox jumps over",
    "In a world where technology",
    "Once upon a time, there was",
    "The capital city of Japan is",
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
                max_tokens=50,
                temperature=1,
                top_k=64,
                echo_back=True
            )
        
        # Decode to text
        generated_text = tokenizer.decode(generated_ids)
        
        # Print results
        print("\n-----")
        print(f"Input: {text}")
        print(f"Generated: {generated_text}")
        print("-----")

load_model_and_generate("/home/xr/code/projects/llm/nanoGPT_tests/checkpoints/checkpoint_fineweb_1B_tied_999735296/checkpoint_fineweb_1B_tied_999735296.pt")
