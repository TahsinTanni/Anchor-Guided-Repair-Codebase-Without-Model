import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load WikiText-2 test set
print("\nLoading WikiText-2 test dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
calibration_text = "\n\n".join(dataset["text"])
print(f"Loaded WikiText-2 test set with {len(dataset)} samples")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load the baseline FP16 model (faster)
print("\n" + "="*60)
print("Loading Qwen2.5-0.5B model in FP16...")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_baseline = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16 for speed
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)

print(f"Model loaded on {model_baseline.device}")

# Function to calculate perplexity (optimized)
def calculate_perplexity(model, tokenizer, text, desc=""):
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = 2048  # Reduced for memory efficiency
    stride = 512  # Smaller stride for better coverage
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

# Measure baseline perplexity (PPL0)
print("\n" + "="*60)
print("Measuring Baseline Perplexity (PPL0) on WikiText-2...")
print("="*60)
PPL0 = calculate_perplexity(model_baseline, tokenizer, calibration_text, "Baseline")
print(f"Baseline PPL0: {PPL0:.4f}")

# Function to inject noise
def inject_weight_noise(model, sigma):
    original_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.device.type != 'meta':
            original_weights[name] = param.data.clone()
            noise = torch.randn_like(param) * sigma
            param.data.add_(noise)
    return original_weights

# Function to restore weights
def restore_weights(model, original_weights):
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])

# Use fewer sigma candidates for speed (4 instead of 6)
sigma_candidates = np.logspace(-5, -2, 4)
print(f"\n" + "="*60)
print(f"Testing {len(sigma_candidates)} candidate σ values...")
print(f"σ candidates: {sigma_candidates}")
print("="*60)

results = []

for i, sigma in enumerate(sigma_candidates):
    print(f"\n[{i+1}/{len(sigma_candidates)}] Testing σ = {sigma:.6f}")
    
    original_weights = inject_weight_noise(model_baseline, sigma)
    ppl = calculate_perplexity(model_baseline, tokenizer, calibration_text, f"σ={sigma:.6f}")
    restore_weights(model_baseline, original_weights)
    
    r = ppl / PPL0
    print(f"  PPL = {ppl:.4f}, r = {r:.4f}")
    
    results.append({
        'sigma': sigma,
        'ppl': ppl,
        'r': r,
        'log_r': np.log(r)
    })
    
    del original_weights
    torch.cuda.empty_cache()

# Fit quadratic model
sigma_squared = np.array([res['sigma']**2 for res in results])
log_r = np.array([res['log_r'] for res in results])
coeffs = np.polyfit(sigma_squared, log_r, deg=1)
a = coeffs[0]
b = coeffs[1]

print(f"\n" + "="*60)
print(f"Quadratic Model Fitting Results (WikiText-2)")
print("="*60)
print(f"Fitted parameter 'a': {a:.4f}")
print(f"Intercept 'b': {b:.6f}")
print(f"Model: r ≈ exp({a:.4f} * σ²)")

# Compute optimal sigmas
target_r_values = [1.05, 1.10, 1.15, 1.20]
print(f"\n" + "="*60)
print(f"Optimal σ* for Target PPL Increases (WikiText-2)")
print("="*60)

optimal_sigmas = {}
for target_r in target_r_values:
    if a > 0:
        sigma_star = np.sqrt(np.log(target_r) / a)
        pct = int((target_r-1)*100)
        optimal_sigmas[f"{pct}%"] = sigma_star
        target_ppl = PPL0 * target_r
        print(f"  {pct}% PPL increase → σ* = {sigma_star:.6f} (target PPL={target_ppl:.4f})")

# Verification
target_r_verify = 1.10
if a > 0:
    sigma_verify = np.sqrt(np.log(target_r_verify) / a)
    
    print(f"\n" + "="*60)
    print(f"Verification: Testing σ* = {sigma_verify:.6f} for 10% PPL increase")
    print("="*60)
    
    original_weights_verify = inject_weight_noise(model_baseline, sigma_verify)
    ppl_verify = calculate_perplexity(model_baseline, tokenizer, calibration_text, "Verification")
    r_verify = ppl_verify / PPL0
    restore_weights(model_baseline, original_weights_verify)
    
    print(f"\nVerification Results:")
    print(f"  Target r: {target_r_verify:.4f}, Actual r: {r_verify:.4f}")
    print(f"  Target PPL: {PPL0 * target_r_verify:.4f}, Actual PPL: {ppl_verify:.4f}")
    print(f"  Error: {abs(r_verify - target_r_verify)/target_r_verify * 100:.2f}%")
    
    # Save results
    detailed_results = {
        'dataset': 'wikitext2_test',
        'num_samples': len(dataset),
        'baseline_ppl': PPL0,
        'sigma_candidates': sigma_candidates.tolist(),
        'candidate_results': results,
        'fitted_parameter_a': a,
        'fitted_intercept_b': b,
        'optimal_sigmas': optimal_sigmas,
        'verification': {
            'sigma': sigma_verify,
            'target_r': target_r_verify,
            'actual_r': r_verify,
            'target_ppl': PPL0 * target_r_verify,
            'actual_ppl': ppl_verify,
            'error_percent': abs(r_verify - target_r_verify)/target_r_verify * 100
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open("weight_noise_injection_wikitext2_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=4)
    
    print(f"\n✓ Results saved to: weight_noise_injection_wikitext2_results.json")
    
    with open("weight_noise_injection_wikitext2_summary.txt", 'w', encoding='utf-8') as f:
        f.write("Weight Noise Injection Analysis Summary (WikiText-2)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: WikiText-2 Test Set ({len(dataset)} samples)\n")
        f.write(f"Baseline PPL0: {PPL0:.4f}\n\n")
        f.write("Fitted Quadratic Model:\n")
        f.write(f"  Parameter a: {a:.4f}\n")
        f.write(f"  Model: r ~= exp({a:.4f} * sigma^2)\n\n")
        f.write("Optimal sigma* for Target PPL Increases:\n")
        for target, sigma_opt in optimal_sigmas.items():
            f.write(f"  {target}: sigma* = {sigma_opt:.6f}\n")
        f.write(f"\nVerification (10% PPL increase):\n")
        f.write(f"  sigma*: {sigma_verify:.6f}\n")
        f.write(f"  Target PPL: {PPL0 * target_r_verify:.4f}\n")
        f.write(f"  Actual PPL: {ppl_verify:.4f}\n")
        f.write(f"  Error: {abs(r_verify - target_r_verify)/target_r_verify * 100:.2f}%\n")
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✓ Summary saved to: weight_noise_injection_wikitext2_summary.txt")

print("\n" + "="*60)
print("✓ Weight Noise Injection Analysis Complete (WikiText-2)!")
print("="*60)
