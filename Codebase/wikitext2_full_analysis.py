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

# Load WikiText-2 optimal sigmas
print("\nLoading WikiText-2 optimal sigma values...")
with open("weight_noise_injection_wikitext2_results.json", 'r') as f:
    noise_results = json.load(f)

optimal_sigmas = noise_results['optimal_sigmas']
print(f"\nOptimal sigmas for WikiText-2:")
for pct, sigma in optimal_sigmas.items():
    print(f"  {pct} PPL increase → σ = {sigma:.6f}")

# Function to calculate perplexity (optimized with FP16)
def calculate_perplexity(model, tokenizer, text, desc=""):
    """Calculate perplexity using FP16 for memory efficiency."""
    print(f"\nCalculating perplexity for {desc}...")
    
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = 2048  # Reduced from max_position_embeddings to avoid OOM
    stride = 512  # Reduced stride for better coverage
    seq_len = encodings.input_ids.size(1)
    
    print(f"  Tokens: {seq_len}, Iterations: ~{(seq_len - max_length) // stride + 1}")
    
    nlls = []
    prev_end_loc = 0
    
    model.eval()
    
    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc=desc):
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
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    perplexity_value = ppl.item()
    
    print(f"  Perplexity: {perplexity_value:.2f}")
    
    return perplexity_value

# Function to inject weight noise
def inject_weight_noise(model, sigma):
    """Inject Gaussian noise N(0, σ²) into all model parameters IN-PLACE."""
    print(f"  Injecting Gaussian noise (σ={sigma:.6f})...")
    noisy_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.device.type != 'meta':
            noise = torch.randn_like(param) * sigma
            param.data.add_(noise)
            noisy_count += 1
    
    print(f"  ✓ Injected noise into {noisy_count} parameters")

# Function to save model
def save_model(model, tokenizer, save_path, description):
    """Save model and tokenizer."""
    print(f"  Saving {description} to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  ✓ Saved")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Noise levels to test
target_levels = ['5%', '10%', '15%', '20%']

# ==================== SAVE CLEAN MODELS FIRST ====================
print("\n" + "="*70)
print("STEP 1: SAVING CLEAN MODELS")
print("="*70)

print("\n[1/3] Clean FP16 Model")
model_clean_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)
ppl_clean_fp16 = calculate_perplexity(model_clean_fp16, tokenizer, text, "Clean FP16")
save_model(model_clean_fp16, tokenizer, "saved_models/wikitext2_clean_fp16", "Clean FP16")
del model_clean_fp16
torch.cuda.empty_cache()

print("\n[2/3] Clean INT8 Model")
model_clean_int8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)
ppl_clean_int8 = calculate_perplexity(model_clean_int8, tokenizer, text, "Clean INT8")
del model_clean_int8
torch.cuda.empty_cache()

print("\n[3/3] Clean 4-BIT Model")
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model_clean_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_4bit,
    device_map="auto",
    trust_remote_code=True
)
ppl_clean_4bit = calculate_perplexity(model_clean_4bit, tokenizer, text, "Clean 4-BIT")
del model_clean_4bit
torch.cuda.empty_cache()

print("\n✓ All clean models processed")

# Set baseline values
print("\nUsing calculated clean model perplexities...")
baseline_fp16_ppl = ppl_clean_fp16
baseline_int8_ppl = ppl_clean_int8
baseline_4bit_ppl = ppl_clean_4bit

print(f"\nBaseline Perplexities (Clean Models - WikiText-2):")
print(f"  FP16: {baseline_fp16_ppl:.2f}")
print(f"  INT8: {baseline_int8_ppl:.2f}")
print(f"  4-BIT: {baseline_4bit_ppl:.2f}")

# Results storage
all_results = {
    'dataset': 'wikitext2_test',
    'num_samples': len(dataset),
    'model_name': model_name,
    'baseline': {
        'fp16': baseline_fp16_ppl,
        'int8': baseline_int8_ppl,
        '4bit': baseline_4bit_ppl
    },
    'noise_levels': {},
    'optimal_sigmas_source': 'wikitext2_calculated',
    'timestamp': datetime.now().isoformat()
}

# ==================== PROCESS EACH NOISE LEVEL ====================
for target_pct in target_levels:
    sigma = optimal_sigmas[target_pct]
    pct = int(target_pct.rstrip('%'))
    
    print("\n" + "="*70)
    print(f"STEP 2: PROCESSING {pct}% NOISE LEVEL (σ={sigma:.6f})")
    print("="*70)
    
    level_results = {
        'target_ppl_increase': pct,
        'sigma': sigma,
        'results': {}
    }
    
    # ==================== FP16 NOISY MODEL ====================
    print(f"\n[1/3] FP16 with {pct}% Noise")
    print("-" * 60)
    
    # Load FP16, inject noise, calculate perplexity, save
    print("Loading model in FP16...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager'
    )
    
    inject_weight_noise(model_fp16, sigma)
    
    # Calculate perplexity
    ppl_noisy_fp16 = calculate_perplexity(model_fp16, tokenizer, text, f"Noisy FP16 ({pct}%)")
    
    degradation_fp16 = ((ppl_noisy_fp16 - baseline_fp16_ppl) / baseline_fp16_ppl) * 100
    print(f"  Degradation vs Clean FP16: {degradation_fp16:+.2f}%")
    
    level_results['results']['fp16'] = {
        'perplexity': ppl_noisy_fp16,
        'degradation_vs_clean_fp16_pct': degradation_fp16
    }
    
    # Save noisy FP16 model
    save_model(model_fp16, tokenizer, f"saved_models/wikitext2_noisy_fp16_{pct}pct",
               f"Noisy FP16 ({pct}%)")
    
    del model_fp16
    torch.cuda.empty_cache()
    
    # ==================== INT8 NOISY MODEL ====================
    print(f"\n[2/3] INT8 Quantized with {pct}% Noise")
    print("-" * 60)
    
    # Load FP16, inject noise, save noisy FP16, then quantize to INT8
    print("Loading model in FP16...")
    model_for_int8 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager'
    )
    
    inject_weight_noise(model_for_int8, sigma)
    
    # Save FP16 before INT8 quantization
    save_model(model_for_int8, tokenizer, f"saved_models/wikitext2_noisy_fp16_before_int8_{pct}pct",
               f"Noisy FP16 before INT8 ({pct}%)")
    
    del model_for_int8
    torch.cuda.empty_cache()
    
    # Now load the noisy FP16 model and quantize it to INT8
    print("Quantizing noisy model to INT8...")
    model_noisy_int8 = AutoModelForCausalLM.from_pretrained(
        f"saved_models/wikitext2_noisy_fp16_before_int8_{pct}pct",
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Calculate perplexity
    ppl_noisy_int8 = calculate_perplexity(model_noisy_int8, tokenizer, text, f"Noisy INT8 ({pct}%)")
    
    degradation_int8 = ((ppl_noisy_int8 - baseline_int8_ppl) / baseline_int8_ppl) * 100
    print(f"  Degradation vs Clean INT8: {degradation_int8:+.2f}%")
    
    level_results['results']['int8'] = {
        'perplexity': ppl_noisy_int8,
        'degradation_vs_clean_int8_pct': degradation_int8,
        'degradation_vs_clean_fp16_pct': ((ppl_noisy_int8 - baseline_fp16_ppl) / baseline_fp16_ppl) * 100
    }
    
    del model_noisy_int8
    torch.cuda.empty_cache()
    
    # ==================== 4-BIT NOISY MODEL ====================
    print(f"\n[3/3] 4-BIT Quantized with {pct}% Noise")
    print("-" * 60)
    
    # Load FP16, inject noise, save noisy FP16, then quantize to 4-BIT
    print("Loading model in FP16...")
    model_for_4bit = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager'
    )
    
    inject_weight_noise(model_for_4bit, sigma)
    
    # Save noisy FP16 before 4-BIT quantization
    save_model(model_for_4bit, tokenizer, f"saved_models/wikitext2_noisy_fp16_before_4bit_{pct}pct",
               f"Noisy FP16 before 4-BIT ({pct}%)")
    
    del model_for_4bit
    torch.cuda.empty_cache()
    
    # Now load the noisy FP16 model and quantize it to 4-BIT
    print("Quantizing noisy model to 4-BIT...")
    bnb_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model_noisy_4bit = AutoModelForCausalLM.from_pretrained(
        f"saved_models/wikitext2_noisy_fp16_before_4bit_{pct}pct",
        quantization_config=bnb_config_4bit,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Calculate perplexity
    ppl_noisy_4bit = calculate_perplexity(model_noisy_4bit, tokenizer, text, f"Noisy 4-BIT ({pct}%)")
    
    degradation_4bit = ((ppl_noisy_4bit - baseline_4bit_ppl) / baseline_4bit_ppl) * 100
    print(f"  Degradation vs Clean 4-BIT: {degradation_4bit:+.2f}%")
    
    level_results['results']['4bit'] = {
        'perplexity': ppl_noisy_4bit,
        'degradation_vs_clean_4bit_pct': degradation_4bit,
        'degradation_vs_clean_fp16_pct': ((ppl_noisy_4bit - baseline_fp16_ppl) / baseline_fp16_ppl) * 100
    }
    
    del model_noisy_4bit
    torch.cuda.empty_cache()
    
    # Store level results
    all_results['noise_levels'][target_pct] = level_results
    
    print(f"\n✓ Completed {pct}% noise level")

# ==================== SUMMARY ====================
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print("\nBaseline (Clean Models):")
print(f"  FP16: {baseline_fp16_ppl:.2f}")
print(f"  INT8: {baseline_int8_ppl:.2f}")
print(f"  4-BIT: {baseline_4bit_ppl:.2f}")

for target_pct in target_levels:
    level_data = all_results['noise_levels'][target_pct]
    pct = level_data['target_ppl_increase']
    sigma = level_data['sigma']
    
    print(f"\n{pct}% Noise Target (σ={sigma:.6f}):")
    
    fp16_data = level_data['results']['fp16']
    int8_data = level_data['results']['int8']
    bit4_data = level_data['results']['4bit']
    
    print(f"  FP16:  {fp16_data['perplexity']:.2f} ({fp16_data['degradation_vs_clean_fp16_pct']:+.2f}% vs clean FP16)")
    print(f"  INT8:  {int8_data['perplexity']:.2f} ({int8_data['degradation_vs_clean_int8_pct']:+.2f}% vs clean INT8)")
    print(f"  4-BIT: {bit4_data['perplexity']:.2f} ({bit4_data['degradation_vs_clean_4bit_pct']:+.2f}% vs clean 4-BIT)")

# Save results
output_json = "noisy_quantized_perplexity_wikitext2.json"
with open(output_json, 'w') as f:
    json.dump(all_results, f, indent=4)
print(f"\n✓ Results saved to: {output_json}")

# Save summary
output_txt = "noisy_quantized_perplexity_wikitext2.txt"
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("Weight-Noisy Quantized Model Analysis (WikiText-2)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: WikiText-2 Test ({len(dataset)} samples)\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("BASELINE (Clean Models)\n")
    f.write("-"*70 + "\n")
    f.write(f"FP16:  {baseline_fp16_ppl:.2f}\n")
    f.write(f"INT8:  {baseline_int8_ppl:.2f}\n")
    f.write(f"4-BIT: {baseline_4bit_ppl:.2f}\n\n")
    
    for target_pct in target_levels:
        level_data = all_results['noise_levels'][target_pct]
        pct = level_data['target_ppl_increase']
        sigma = level_data['sigma']
        
        fp16_data = level_data['results']['fp16']
        int8_data = level_data['results']['int8']
        bit4_data = level_data['results']['4bit']
        
        f.write(f"\nNOISY MODELS - {pct}% Target (σ={sigma:.6f})\n")
        f.write("-"*70 + "\n")
        f.write(f"FP16:  {fp16_data['perplexity']:.2f} ({fp16_data['degradation_vs_clean_fp16_pct']:+.2f}%)\n")
        f.write(f"INT8:  {int8_data['perplexity']:.2f} ({int8_data['degradation_vs_clean_int8_pct']:+.2f}%)\n")
        f.write(f"4-BIT: {bit4_data['perplexity']:.2f} ({bit4_data['degradation_vs_clean_4bit_pct']:+.2f}%)\n")

print(f"✓ Summary saved to: {output_txt}")

print("\n" + "="*70)
print("SAVED MODELS")
print("="*70)
print("\nClean Models:")
print("  - saved_models/wikitext2_clean_fp16/")
print("  - Clean INT8 and 4-BIT models are quantized on-the-fly (not saved)")

print("\nNoisy Models:")
for pct_str in target_levels:
    pct = int(pct_str.rstrip('%'))
    print(f"\n  {pct}% Noise Level:")
    print(f"    - saved_models/wikitext2_noisy_fp16_{pct}pct/")
    print(f"    - saved_models/wikitext2_noisy_fp16_before_int8_{pct}pct/")
    print(f"    - saved_models/wikitext2_noisy_fp16_before_4bit_{pct}pct/")
    print(f"    - Noisy INT8 and 4-BIT models are quantized on-the-fly from saved FP16 models")

print("\n" + "="*70)
print("✓ ALL ANALYSIS COMPLETE!")
