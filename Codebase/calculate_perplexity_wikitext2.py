import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load WikiText-2 test set
print("\nLoading WikiText-2 test dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
print(f"Loaded WikiText-2 test set with {len(dataset)} samples")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text, quantization_type):
    print(f"\n{'='*60}")
    print(f"Calculating perplexity for {quantization_type}...")
    print(f"{'='*60}")
    
    # Tokenize the text
    print("Tokenizing text...")
    encodings = tokenizer(text, return_tensors="pt")
    
    # Calculate perplexity
    max_length = 2048  # Reduced from model.config.max_position_embeddings for memory efficiency
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    print(f"Sequence length: {seq_len} tokens")
    print(f"Max length: {max_length}")
    print(f"Stride: {stride}")
    
    nlls = []
    prev_end_loc = 0
    
    model.eval()
    
    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Processing"):
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
    
    return perplexity_value, seq_len

# Store all results
all_results = {}

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ============= FP16 (Half Precision) =============
print("\n" + "="*60)
print("Loading Qwen2.5-0.5B model with FP16...")
print("="*60)

model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    _attn_implementation_internal='eager'
)

print(f"FP16 Model loaded on {model_fp16.device}")
print(f"Model dtype: {next(model_fp16.parameters()).dtype}")

# Calculate FP16 perplexity
perplexity_fp16, seq_len = calculate_perplexity(model_fp16, tokenizer, text, "FP16")

# Save FP16 results
results_fp16 = {
    "model_name": model_name,
    "quantization": "fp16",
    "perplexity": perplexity_fp16,
    "dataset": "wikitext2_test",
    "num_samples": len(dataset),
    "sequence_length": seq_len,
    "device": str(device),
    "timestamp": datetime.now().isoformat(),
    "num_parameters": sum(p.numel() for p in model_fp16.parameters()),
    "dtype": "float16"
}

all_results["fp16"] = results_fp16

# Save FP16 results to file
with open("qwen2.5_0.5b_perplexity_fp16_wikitext2.json", 'w') as f:
    json.dump(results_fp16, f, indent=4)

print(f"\nFP16 results saved to: qwen2.5_0.5b_perplexity_fp16_wikitext2.json")

# Clear FP16 model from memory
del model_fp16
torch.cuda.empty_cache()

# ============= INT8 QUANTIZATION =============
print("\n" + "="*60)
print("Loading Qwen2.5-0.5B model with INT8 quantization...")
print("="*60)

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
perplexity_int8, _ = calculate_perplexity(model_int8, tokenizer, text, "INT8")

# Save INT8 results
results_int8 = {
    "model_name": model_name,
    "quantization": "int8",
    "perplexity": perplexity_int8,
    "dataset": "wikitext2_test",
    "num_samples": len(dataset),
    "sequence_length": seq_len,
    "device": str(device),
    "timestamp": datetime.now().isoformat(),
    "num_parameters": sum(p.numel() for p in model_int8.parameters())
}

all_results["int8"] = results_int8

# Save INT8 results to file
with open("qwen2.5_0.5b_perplexity_int8_wikitext2.json", 'w') as f:
    json.dump(results_int8, f, indent=4)

print(f"\nINT8 results saved to: qwen2.5_0.5b_perplexity_int8_wikitext2.json")

# Clear INT8 model from memory
del model_int8
torch.cuda.empty_cache()

# ============= 4-BIT QUANTIZATION =============
print("\n" + "="*60)
print("Loading Qwen2.5-0.5B model with 4-bit quantization...")
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
perplexity_4bit, _ = calculate_perplexity(model_4bit, tokenizer, text, "4-BIT")

# Save 4-bit results
results_4bit = {
    "model_name": model_name,
    "quantization": "4bit (NF4)",
    "perplexity": perplexity_4bit,
    "dataset": "wikitext2_test",
    "num_samples": len(dataset),
    "sequence_length": seq_len,
    "device": str(device),
    "timestamp": datetime.now().isoformat(),
    "num_parameters": sum(p.numel() for p in model_4bit.parameters())
}

all_results["4bit"] = results_4bit

# Save 4-bit results to file
with open("qwen2.5_0.5b_perplexity_4bit_wikitext2.json", 'w') as f:
    json.dump(results_4bit, f, indent=4)

print(f"\n4-bit results saved to: qwen2.5_0.5b_perplexity_4bit_wikitext2.json")

# ============= SUMMARY =============
print("\n" + "="*60)
print("PERPLEXITY COMPARISON SUMMARY (WikiText-2)")
print("="*60)

print(f"FP16 Perplexity:  {perplexity_fp16:.2f}")
print(f"INT8 Perplexity:  {perplexity_int8:.2f}")
print(f"4-BIT Perplexity: {perplexity_4bit:.2f}")
print("="*60)

# Calculate degradation
int8_degradation = ((perplexity_int8 - perplexity_fp16) / perplexity_fp16) * 100
bit4_degradation = ((perplexity_4bit - perplexity_fp16) / perplexity_fp16) * 100

print("\nPerplexity Increase vs FP16:")
print(f"INT8: {int8_degradation:+.2f}%")
print(f"4-BIT: {bit4_degradation:+.2f}%")
print("="*60)

# Save combined results
with open("qwen2.5_0.5b_perplexity_comparison_wikitext2.json", 'w') as f:
    json.dump(all_results, f, indent=4)

# Save summary to text file
with open("qwen2.5_0.5b_perplexity_comparison_wikitext2.txt", 'w') as f:
    f.write("Qwen2.5-0.5B Perplexity Comparison\n")
    f.write("="*60 + "\n")
    f.write(f"Dataset: WikiText-2 Test Set ({len(dataset)} samples, {seq_len} tokens)\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"FP16 Perplexity:  {perplexity_fp16:.2f}\n")
    f.write(f"INT8 Perplexity:  {perplexity_int8:.2f}\n")
    f.write(f"4-BIT Perplexity: {perplexity_4bit:.2f}\n")
    f.write("="*60 + "\n")
    
    f.write("\nPerplexity Increase vs FP16:\n")
    f.write(f"INT8: {int8_degradation:+.2f}%\n")
    f.write(f"4-BIT: {bit4_degradation:+.2f}%\n")
    f.write("="*60 + "\n")

print("\nComparison results saved to: qwen2.5_0.5b_perplexity_comparison_wikitext2.json")
print("Summary saved to: qwen2.5_0.5b_perplexity_comparison_wikitext2.txt")

print("\n" + "="*60)
print("All perplexity calculations completed!")
print("="*60)
