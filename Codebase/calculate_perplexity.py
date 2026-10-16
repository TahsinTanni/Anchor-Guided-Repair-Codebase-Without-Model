import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Phi-3 Mini 3.8B model in FP32
print("\nLoading Phi-3 Mini 3.8B model in FP32...")
model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Using FP32
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    _attn_implementation_internal='eager'
)

print(f"Model loaded on {model.device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Load prompts from prompt_suite_96.json
print("\nLoading prompts from prompt_suite_96.json...")
prompt_file = r"c:\codes2\Robust-Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection\prompt_suite_96.json"
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompt_data = json.load(f)

# Extract all prompts from all categories
all_prompts = []
for category, prompts in prompt_data.items():
    all_prompts.extend(prompts)

print(f"Loaded {len(all_prompts)} prompts from {len(prompt_data)} categories")

# Combine all prompts into a single text
text = "\n".join(all_prompts)

# Tokenize the text
print("\nTokenizing text...")
encodings = tokenizer(text, return_tensors="pt")

# Calculate perplexity
print("\nCalculating perplexity...")
max_length = model.config.max_position_embeddings
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0

model.eval()

with torch.no_grad():
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids, use_cache=False)
        neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

# Calculate perplexity
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
perplexity_value = ppl.item()

print(f"\n{'='*50}")
print(f"Perplexity: {perplexity_value:.2f}")
print(f"{'='*50}")

# Store results
results = {
    "model_name": model_name,
    "perplexity": perplexity_value,
    "dataset": "prompt_suite_96.json",
    "num_prompts": len(all_prompts),
    "num_categories": len(prompt_data),
    "dtype": "float32",
    "device": str(device),
    "timestamp": datetime.now().isoformat(),
    "num_parameters": sum(p.numel() for p in model.parameters()),
    "max_length": max_length,
    "stride": stride
}

# Save to JSON file
output_file = "phi3_perplexity_fp32_prompts.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {output_file}")

# Also save to a simple text file for quick reference
output_txt = "phi3_perplexity_fp32_prompts.txt"
with open(output_txt, 'w') as f:
    f.write(f"Phi-3 Mini 3.8B Perplexity (FP32)\n")
    f.write(f"{'='*50}\n")
    f.write(f"Perplexity: {perplexity_value:.2f}\n")
    f.write(f"Dataset: prompt_suite_96.json\n")
    f.write(f"Number of prompts: {len(all_prompts)}\n")
    f.write(f"Number of categories: {len(prompt_data)}\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"Summary saved to: {output_txt}")
