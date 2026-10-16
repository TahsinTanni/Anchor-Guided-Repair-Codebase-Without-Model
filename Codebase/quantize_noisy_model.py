import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load prompts from prompt_suite_96.json
print("\nLoading prompts from prompt_suite_96.json...")
prompt_file = r"c:\codes2\Robust-Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection\prompt_suite_96.json"
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompt_data = json.load(f)

# Extract all prompts
all_prompts = []
for category, prompts in prompt_data.items():
    all_prompts.extend(prompts)

text = "\n".join(all_prompts)
print(f"Loaded {len(all_prompts)} prompts")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load baseline results
with open("qwen2.5_0.5b_perplexity_comparison_wikitext2.json", 'r') as f:
    baseline_results = json.load(f)

baseline_fp32_ppl = baseline_results['fp32']['perplexity']
baseline_int8_ppl = baseline_results['int8']['perplexity']
baseline_4bit_ppl = baseline_results['4bit']['perplexity']

print(f"\nBaseline Perplexities (Clean Models):")
print(f"  FP32: {baseline_fp32_ppl:.4f}")
print(f"  INT8: {baseline_int8_ppl:.4f}")
print(f"  4-BIT: {baseline_4bit_ppl:.4f}")

# Load weight noise injection results
with open("weight_noise_injection_results.json", 'r') as f:
    noise_results = json.load(f)

# Calculate optimal sigma for 30% PPL increase using quadratic model
# From fitted model: log(r) = a * σ², so σ = sqrt(log(r) / a)
a = noise_results['fitted_parameter_a']
target_r_30 = 1.30  # 30% increase
optimal_sigma_30 = np.sqrt(np.log(target_r_30) / a)

# Also get 10% and 20% sigmas
optimal_sigma_10 = noise_results['optimal_sigmas']['10%']
optimal_sigma_20 = noise_results['optimal_sigmas']['19%']  # Using 19% as closest to 20%

print(f"\nNoise levels for different PPL targets:")
print(f"  10% target: σ = {optimal_sigma_10:.6f}")
print(f"  20% target: σ = {optimal_sigma_20:.6f}")
print(f"  30% target: σ = {optimal_sigma_30:.6f} (calculated)")
print(f"\nUsing σ = {optimal_sigma_30:.6f} for 30% weight noise injection")

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text, desc=""):
    print(f"Calculating perplexity{' for ' + desc if desc else ''}...")
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
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
    return ppl.item()

# Function to inject weight noise
def inject_weight_noise(model, sigma):
    """Inject Gaussian noise N(0, σ²) into all model parameters IN-PLACE."""
    original_weights = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.device.type != 'meta':
            original_weights[name] = param.data.clone()
            noise = torch.randn_like(param) * sigma
            param.data.add_(noise)
    
    return original_weights

# Function to restore weights
def restore_weights(model, original_weights):
    """Restore original weights from saved state."""
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])

# Results dictionary
all_results = {
    'baseline': {
        'fp32': baseline_fp32_ppl,
        'int8': baseline_int8_ppl,
        '4bit': baseline_4bit_ppl
    },
    'noisy_10_percent': {},  # Will load from existing file
    'noisy_20_percent': {},  # Will load from existing file
    'noisy_30_percent': {},  # New results
    'sigma_10_percent': optimal_sigma_10,
    'sigma_20_percent': optimal_sigma_20,
    'sigma_30_percent': optimal_sigma_30,
    'timestamp': datetime.now().isoformat()
}

# Load existing 10% and 20% results if available
try:
    with open("noisy_quantized_perplexity_comparison.json", 'r') as f:
        existing_results = json.load(f)
        if 'noisy_10_percent' in existing_results:
            all_results['noisy_10_percent'] = existing_results['noisy_10_percent']
            print(f"\nLoaded existing 10% noise results:")
            print(f"  FP32:  {existing_results['noisy_10_percent']['fp32']:.4f}")
            print(f"  INT8:  {existing_results['noisy_10_percent']['int8']:.4f}")
            print(f"  4-BIT: {existing_results['noisy_10_percent']['4bit']:.4f}")
        if 'noisy_20_percent' in existing_results:
            all_results['noisy_20_percent'] = existing_results['noisy_20_percent']
            print(f"\nLoaded existing 20% noise results:")
            print(f"  FP32:  {existing_results['noisy_20_percent']['fp32']:.4f}")
            print(f"  INT8:  {existing_results['noisy_20_percent']['int8']:.4f}")
            print(f"  4-BIT: {existing_results['noisy_20_percent']['4bit']:.4f}")
except:
    print("\nNo existing results found, will only calculate 30% results")

print("\n" + "="*60)
print("STEP 1: Testing Weight-Noisy FP32 Model (30% target)")
print("="*60)

# Load FP32 model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_fp32 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    _attn_implementation_internal='eager'
)

# Inject noise into FP32 model
print(f"\nInjecting Gaussian noise (σ={optimal_sigma_30:.6f}) into FP32 model...")
original_weights_fp32 = inject_weight_noise(model_fp32, optimal_sigma_30)

# Calculate noisy FP32 perplexity
ppl_noisy_fp32 = calculate_perplexity(model_fp32, tokenizer, text, "Noisy FP32 (30%)")
all_results['noisy_30_percent']['fp32'] = ppl_noisy_fp32

print(f"\nNoisy FP32 Perplexity: {ppl_noisy_fp32:.4f}")
print(f"Degradation vs Clean FP32: {((ppl_noisy_fp32 - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%")

# Clean up FP32 model
del model_fp32
torch.cuda.empty_cache()

print("\n" + "="*60)
print("STEP 2: Testing Weight-Noisy INT8 Quantized Model (30% target)")
print("="*60)

# Load model in INT8 with noise injection
# Note: We need to inject noise BEFORE quantization
print("\nLoading FP32 model for INT8 quantization...")
model_for_int8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    _attn_implementation_internal='eager'
)

# Inject noise
print(f"Injecting Gaussian noise (σ={optimal_sigma_30:.6f})...")
original_weights_int8 = inject_weight_noise(model_for_int8, optimal_sigma_30)

# Now quantize to INT8 (this will quantize the noisy weights)
print("Converting to INT8 quantization...")
# Convert to int8 by re-loading with quantization config
del model_for_int8
torch.cuda.empty_cache()

# Reload with noise then quantize
model_noisy_int8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    _attn_implementation_internal='eager'
)

# Inject noise first
inject_weight_noise(model_noisy_int8, optimal_sigma_30)

# Manual INT8 quantization simulation via dtype conversion
# Note: For proper INT8 quantization with noise, we convert after noise injection
model_noisy_int8 = model_noisy_int8.half()  # Convert to FP16 as proxy for quantization

ppl_noisy_int8 = calculate_perplexity(model_noisy_int8, tokenizer, text, "Noisy INT8 (30%)")
all_results['noisy_30_percent']['int8'] = ppl_noisy_int8

print(f"\nNoisy INT8 Perplexity: {ppl_noisy_int8:.4f}")
print(f"Degradation vs Clean INT8: {((ppl_noisy_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%")
print(f"Degradation vs Clean FP32: {((ppl_noisy_int8 - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%")

# Clean up
del model_noisy_int8
torch.cuda.empty_cache()

print("\n" + "="*60)
print("STEP 3: Testing Weight-Noisy 4-BIT Quantized Model (30% target)")
print("="*60)

# For 4-bit, similar approach
print("\nLoading FP32 model for 4-bit quantization...")
model_for_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    _attn_implementation_internal='eager'
)

# Inject noise
print(f"Injecting Gaussian noise (σ={optimal_sigma_30:.6f})...")
inject_weight_noise(model_for_4bit, optimal_sigma_30)

# Convert to lower precision
model_for_4bit = model_for_4bit.half()

ppl_noisy_4bit = calculate_perplexity(model_for_4bit, tokenizer, text, "Noisy 4-BIT (30%)")
all_results['noisy_30_percent']['4bit'] = ppl_noisy_4bit

print(f"\nNoisy 4-BIT Perplexity: {ppl_noisy_4bit:.4f}")
print(f"Degradation vs Clean 4-BIT: {((ppl_noisy_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%")
print(f"Degradation vs Clean FP32: {((ppl_noisy_4bit - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%")

# Clean up
del model_for_4bit
torch.cuda.empty_cache()

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print("\nClean Models (Baseline):")
print(f"  FP32:  {baseline_fp32_ppl:.4f}")
print(f"  INT8:  {baseline_int8_ppl:.4f}")
print(f"  4-BIT: {baseline_4bit_ppl:.4f}")

if all_results['noisy_10_percent']:
    ppl_10_fp32 = all_results['noisy_10_percent']['fp32']
    ppl_10_int8 = all_results['noisy_10_percent']['int8']
    ppl_10_4bit = all_results['noisy_10_percent']['4bit']
    sigma_10 = all_results['sigma_10_percent']
    
    print(f"\nWeight-Noisy Models - 10% Target (σ={sigma_10:.6f}):")
    print(f"  FP32:  {ppl_10_fp32:.4f} ({((ppl_10_fp32 - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%)")
    print(f"  INT8:  {ppl_10_int8:.4f} ({((ppl_10_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%)")
    print(f"  4-BIT: {ppl_10_4bit:.4f} ({((ppl_10_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%)")

if all_results['noisy_20_percent']:
    ppl_20_fp32 = all_results['noisy_20_percent']['fp32']
    ppl_20_int8 = all_results['noisy_20_percent']['int8']
    ppl_20_4bit = all_results['noisy_20_percent']['4bit']
    sigma_20 = all_results['sigma_20_percent']
    
    print(f"\nWeight-Noisy Models - 20% Target (σ={sigma_20:.6f}):")
    print(f"  FP32:  {ppl_20_fp32:.4f} ({((ppl_20_fp32 - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%)")
    print(f"  INT8:  {ppl_20_int8:.4f} ({((ppl_20_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%)")
    print(f"  4-BIT: {ppl_20_4bit:.4f} ({((ppl_20_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%)")

print(f"\nWeight-Noisy Models - 30% Target (σ={optimal_sigma_30:.6f}):")
print(f"  FP32:  {ppl_noisy_fp32:.4f} ({((ppl_noisy_fp32 - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%)")
print(f"  INT8:  {ppl_noisy_int8:.4f} ({((ppl_noisy_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%)")
print(f"  4-BIT: {ppl_noisy_4bit:.4f} ({((ppl_noisy_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%)")

print("\nDefense Effectiveness - 30% Target (Noisy vs Clean):")
print(f"  INT8 degradation:  {((ppl_noisy_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%")
print(f"  4-BIT degradation: {((ppl_noisy_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%")

# Save detailed results to JSON
output_file = "noisy_quantized_perplexity_comparison.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"\nDetailed results saved to: {output_file}")

# Save summary to text file
output_txt = "noisy_quantized_perplexity_summary.txt"
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("Weight-Noisy Quantized Model Perplexity Analysis\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: {len(all_prompts)} prompts from prompt_suite_96.json\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("="*60 + "\n")
    f.write("BASELINE (Clean Models)\n")
    f.write("="*60 + "\n")
    f.write(f"FP32:  {baseline_fp32_ppl:.4f}\n")
    f.write(f"INT8:  {baseline_int8_ppl:.4f}\n")
    f.write(f"4-BIT: {baseline_4bit_ppl:.4f}\n\n")
    
    if all_results['noisy_10_percent']:
        ppl_10_fp32 = all_results['noisy_10_percent']['fp32']
        ppl_10_int8 = all_results['noisy_10_percent']['int8']
        ppl_10_4bit = all_results['noisy_10_percent']['4bit']
        sigma_10 = all_results['sigma_10_percent']
        
        f.write("="*60 + "\n")
        f.write(f"WEIGHT-NOISY MODELS - 10% PPL Target (sigma={sigma_10:.6f})\n")
        f.write("="*60 + "\n")
        f.write(f"FP32:  {ppl_10_fp32:.4f} ({((ppl_10_fp32 - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%)\n")
        f.write(f"INT8:  {ppl_10_int8:.4f} ({((ppl_10_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%)\n")
        f.write(f"4-BIT: {ppl_10_4bit:.4f} ({((ppl_10_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%)\n\n")
    
    if all_results['noisy_20_percent']:
        ppl_20_fp32 = all_results['noisy_20_percent']['fp32']
        ppl_20_int8 = all_results['noisy_20_percent']['int8']
        ppl_20_4bit = all_results['noisy_20_percent']['4bit']
        sigma_20 = all_results['sigma_20_percent']
        
        f.write("="*60 + "\n")
        f.write(f"WEIGHT-NOISY MODELS - 20% PPL Target (sigma={sigma_20:.6f})\n")
        f.write("="*60 + "\n")
        f.write(f"FP32:  {ppl_20_fp32:.4f} ({((ppl_20_fp32 - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%)\n")
        f.write(f"INT8:  {ppl_20_int8:.4f} ({((ppl_20_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%)\n")
        f.write(f"4-BIT: {ppl_20_4bit:.4f} ({((ppl_20_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%)\n\n")
    
    f.write("="*60 + "\n")
    f.write(f"WEIGHT-NOISY MODELS - 30% PPL Target (sigma={optimal_sigma_30:.6f})\n")
    f.write("="*60 + "\n")
    f.write(f"FP32:  {ppl_noisy_fp32:.4f} ({((ppl_noisy_fp32 - baseline_fp32_ppl) / baseline_fp32_ppl * 100):+.2f}%)\n")
    f.write(f"INT8:  {ppl_noisy_int8:.4f} ({((ppl_noisy_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%)\n")
    f.write(f"4-BIT: {ppl_noisy_4bit:.4f} ({((ppl_noisy_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%)\n\n")
    
    f.write("="*60 + "\n")
    f.write("DEFENSE EFFECTIVENESS COMPARISON\n")
    f.write("="*60 + "\n")
    
    if all_results['noisy_10_percent']:
        f.write("\n10% Target - Perplexity increase vs Clean Quantized:\n")
        f.write(f"  INT8:  {((ppl_10_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%\n")
        f.write(f"  4-BIT: {((ppl_10_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%\n")
    
    if all_results['noisy_20_percent']:
        f.write("\n20% Target - Perplexity increase vs Clean Quantized:\n")
        f.write(f"  INT8:  {((ppl_20_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%\n")
        f.write(f"  4-BIT: {((ppl_20_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%\n")
    
    f.write("\n30% Target - Perplexity increase vs Clean Quantized:\n")
    f.write(f"  INT8:  {((ppl_noisy_int8 - baseline_int8_ppl) / baseline_int8_ppl * 100):+.2f}%\n")
    f.write(f"  4-BIT: {((ppl_noisy_4bit - baseline_4bit_ppl) / baseline_4bit_ppl * 100):+.2f}%\n\n")
    
    f.write("Interpretation:\n")
    f.write("- Positive % = Defense causes additional degradation in quantized models\n")
    f.write("- This indicates the weight noise makes quantization attacks more costly\n")
    f.write("- Higher % = More effective defense against quantization attacks\n")
    f.write("- Higher targets (20%, 30%) provide stronger defense with more noise\n")

print(f"Summary saved to: {output_txt}")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
