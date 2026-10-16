import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

model_name = "microsoft/Phi-3-mini-4k-instruct"

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text, quantization_type):
    print(f"\n{'='*60}")
    print(f"Calculating perplexity for {quantization_type}...")
    print(f"{'='*60}")
    
    # Tokenize the text
    print("Tokenizing text...")
    encodings = tokenizer(text, return_tensors="pt")
    
    # Calculate perplexity
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
    
    print(f"\n{'='*60}")
    print(f"{quantization_type} Perplexity: {perplexity_value:.2f}")
    print(f"{'='*60}")
    
    return perplexity_value

# Store all results
all_results = {}

# ============= INT8 QUANTIZATION =============
print("\n" + "="*60)
print("Loading Phi-3 Mini 3.8B model with INT8 quantization...")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

quantization_config_int8 = BitsAndBytesConfig(load_in_8bit=True)

model_int8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config_int8,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)

print(f"INT8 Model loaded on {model_int8.device}")

# Calculate INT8 perplexity
perplexity_int8 = calculate_perplexity(model_int8, tokenizer, text, "INT8")

# Save INT8 results
results_int8 = {
    "model_name": model_name,
    "quantization": "int8",
    "perplexity": perplexity_int8,
    "dataset": "prompt_suite_96.json",
    "num_prompts": len(all_prompts),
    "num_categories": len(prompt_data),
    "device": str(device),
    "timestamp": datetime.now().isoformat(),
    "num_parameters": sum(p.numel() for p in model_int8.parameters())
}

all_results["int8"] = results_int8

# Save INT8 results to file
with open("phi3_perplexity_int8_prompts.json", 'w') as f:
    json.dump(results_int8, f, indent=4)

print(f"\nINT8 results saved to: phi3_perplexity_int8_prompts.json")

# Clear INT8 model from memory
del model_int8
torch.cuda.empty_cache()

# ============= 4-BIT QUANTIZATION =============
print("\n" + "="*60)
print("Loading Phi-3 Mini 3.8B model with 4-bit quantization...")
print("="*60)

quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config_4bit,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)

print(f"4-bit Model loaded on {model_4bit.device}")

# Calculate 4-bit perplexity
perplexity_4bit = calculate_perplexity(model_4bit, tokenizer, text, "4-BIT")

# Save 4-bit results
results_4bit = {
    "model_name": model_name,
    "quantization": "4bit (NF4)",
    "perplexity": perplexity_4bit,
    "dataset": "prompt_suite_96.json",
    "num_prompts": len(all_prompts),
    "num_categories": len(prompt_data),
    "device": str(device),
    "timestamp": datetime.now().isoformat(),
    "num_parameters": sum(p.numel() for p in model_4bit.parameters())
}

all_results["4bit"] = results_4bit

# Save 4-bit results to file
with open("phi3_perplexity_4bit_prompts.json", 'w') as f:
    json.dump(results_4bit, f, indent=4)

print(f"\n4-bit results saved to: phi3_perplexity_4bit_prompts.json")

# ============= SUMMARY =============
print("\n" + "="*60)
print("PERPLEXITY COMPARISON SUMMARY")
print("="*60)

# Load FP32 results
try:
    with open("phi3_perplexity_fp32_prompts.json", 'r') as f:
        results_fp32 = json.load(f)
        all_results["fp32"] = results_fp32
        print(f"FP32 Perplexity:  {results_fp32['perplexity']:.2f}")
except:
    print("FP32 Perplexity:  Not found")

print(f"INT8 Perplexity:  {perplexity_int8:.2f}")
print(f"4-BIT Perplexity: {perplexity_4bit:.2f}")
print("="*60)

# Save combined results
with open("phi3_perplexity_comparison.json", 'w') as f:
    json.dump(all_results, f, indent=4)

# Save summary to text file
with open("phi3_perplexity_comparison.txt", 'w') as f:
    f.write("Phi-3 Mini 3.8B Perplexity Comparison\n")
    f.write("="*60 + "\n")
    f.write(f"Dataset: prompt_suite_96.json ({len(all_prompts)} prompts)\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    if "fp32" in all_results:
        f.write(f"FP32 Perplexity:  {all_results['fp32']['perplexity']:.2f}\n")
    f.write(f"INT8 Perplexity:  {perplexity_int8:.2f}\n")
    f.write(f"4-BIT Perplexity: {perplexity_4bit:.2f}\n")
    f.write("="*60 + "\n")
    
    # Calculate degradation if FP32 exists
    if "fp32" in all_results:
        fp32_ppl = all_results['fp32']['perplexity']
        int8_degradation = ((perplexity_int8 - fp32_ppl) / fp32_ppl) * 100
        bit4_degradation = ((perplexity_4bit - fp32_ppl) / fp32_ppl) * 100
        
        f.write("\nPerplexity Increase vs FP32:\n")
        f.write(f"INT8: {int8_degradation:+.2f}%\n")
        f.write(f"4-BIT: {bit4_degradation:+.2f}%\n")

print("\nComparison results saved to: phi3_perplexity_comparison.json")
print("Summary saved to: phi3_perplexity_comparison.txt")
