import matplotlib.pyplot as plt
import json

rope = "/home/xr/code/projects/llm/nanoGPT_tests/checkpoints/checkpoint_fineweb_1B_rope_999735296/checkpoint_fineweb_1B_rope_999735296_metrics.json"
default = "/home/xr/code/projects/llm/nanoGPT_tests/checkpoints/checkpoint_fineweb_1B_999948288/checkpoint_fineweb_1B_999948288.json"

with open(rope, 'r') as f:
    data1 = json.load(f)

with open(default, 'r') as f:
    data2 = json.load(f)

data1 = data1["loss"][:2543]
data2 = data2["loss"][:2543]

#make a graph comparing those 2
#name em rope and pe


# get train loss from both
loss1 = data1#[x['train/loss'] for x in data1]
loss2 = data2#[x['train/loss'] for x in data2]

# create plot
plt.figure(figsize=(10,6))
plt.plot(loss1, label='RoPE', alpha=0.8)
plt.plot(loss2, label='PE', alpha=0.8)

plt.title('Training Loss: RoPE vs Positional Encoding')
plt.xlabel('Steps') 
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()