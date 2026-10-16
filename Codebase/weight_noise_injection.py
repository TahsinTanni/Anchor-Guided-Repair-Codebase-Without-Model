import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Extract all prompts from all categories
all_prompts = []
for category, prompts in prompt_data.items():
    all_prompts.extend(prompts)

print(f"Loaded {len(all_prompts)} prompts")

# Use all prompts for analysis
calibration_prompts = all_prompts
calibration_text = "\n".join(calibration_prompts)
print(f"Using all {len(calibration_prompts)} prompts for analysis")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load the baseline FP16 model
print("\n" + "="*60)
print("Loading Qwen2.5-0.5B model in FP16...")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_baseline = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    _attn_implementation_internal='eager'
)

print(f"Model loaded on {model_baseline.device}")
print(f"Model dtype: {next(model_baseline.parameters()).dtype}")

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text, desc=""):
    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    model.eval()
    
    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
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
print("Measuring Baseline Perplexity (PPL0)...")
print("="*60)
PPL0 = calculate_perplexity(model_baseline, tokenizer, calibration_text)
print(f"Baseline PPL0: {PPL0:.4f}")

# Function to inject Gaussian noise into model weights
def inject_weight_noise(model, sigma, record_changes=False):
    """
    Inject Gaussian noise N(0, σ²) into all model parameters IN-PLACE.
    Returns weight changes and original weights for restoration.
    """
    weight_changes = []
    original_weights = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.device.type != 'meta':  # Skip meta tensors
            # Save original weights
            original_weights[name] = param.data.clone()
            
            # Generate Gaussian noise with mean=0, std=sigma
            noise = torch.randn_like(param) * sigma
            
            if record_changes:
                # Record sample changes (first 5 values for brevity)
                original_values = param.data.flatten()[:5].cpu().numpy()
                noisy_values = (param.data.flatten()[:5] + noise.flatten()[:5]).cpu().numpy()
                noise_values = noise.flatten()[:5].cpu().numpy()
                
                weight_changes.append({
                    'layer': name,
                    'shape': list(param.shape),
                    'num_params': param.numel(),
                    'original_values_sample': original_values.tolist(),
                    'noise_values_sample': noise_values.tolist(),
                    'noisy_values_sample': noisy_values.tolist(),
                    'original_mean': param.data.mean().item(),
                    'original_std': param.data.std().item(),
                    'noise_mean': noise.mean().item(),
                    'noise_std': noise.std().item()
                })
            
            # Add noise to parameters IN-PLACE
            param.data.add_(noise)
    
    return original_weights, weight_changes

# Function to restore original weights
def restore_weights(model, original_weights):
    """Restore original weights from saved state."""
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])

# Define candidate sigma values (log-spaced)
sigma_candidates = np.logspace(-5, -2, 6)  # 6 values from 0.00001 to 0.01
print(f"\n" + "="*60)
print(f"Testing {len(sigma_candidates)} candidate σ values...")
print(f"σ candidates: {sigma_candidates}")
print("="*60)

results = []
ppl_values = []

for i, sigma in enumerate(sigma_candidates):
    print(f"\n[{i+1}/{len(sigma_candidates)}] Testing σ = {sigma:.6f}")
    
    # Inject noise and save original weights
    original_weights, _ = inject_weight_noise(model_baseline, sigma, record_changes=False)
    
    # Calculate perplexity
    ppl = calculate_perplexity(model_baseline, tokenizer, calibration_text)
    
    # Restore original weights
    restore_weights(model_baseline, original_weights)
    
    # Calculate PPL increase ratio
    r = ppl / PPL0
    
    print(f"  PPL(σ={sigma:.6f}) = {ppl:.4f}, r = {r:.4f}")
    
    results.append({
        'sigma': sigma,
        'ppl': ppl,
        'r': r,
        'log_r': np.log(r)
    })
    ppl_values.append(ppl)
    
    # Clean up
    del original_weights
    torch.cuda.empty_cache()

# Fit quadratic model: log(r) ≈ a * σ²
# We use: r = exp(a * σ²), so log(r) = a * σ²
sigma_squared = np.array([res['sigma']**2 for res in results])
log_r = np.array([res['log_r'] for res in results])

# Fit a linear model: log_r = a * sigma_squared
# Using polyfit for simple linear regression
coeffs = np.polyfit(sigma_squared, log_r, deg=1)
a = coeffs[0]  # slope
b = coeffs[1]  # intercept (should be close to 0 for theory)

print(f"\n" + "="*60)
print(f"Quadratic Model Fitting Results")
print("="*60)
print(f"Fitted parameter 'a': {a:.4f}")
print(f"Intercept 'b': {b:.6f}")
print(f"Model: log(r) ≈ {a:.4f} * σ² + {b:.6f}")
print(f"Or: r ≈ exp({a:.4f} * σ²)")

# Compute optimal σ* for different target PPL increases
target_r_values = [1.05, 1.10, 1.15, 1.20]  # 5%, 10%, 15%, 20% PPL increase
print(f"\n" + "="*60)
print(f"Optimal σ* for Target PPL Increases")
print("="*60)

optimal_sigmas = {}
for target_r in target_r_values:
    # From: log(r) = a * σ², solve for σ
    # σ* = sqrt(log(r_target) / a)
    if a > 0:
        sigma_star = np.sqrt(np.log(target_r) / a)
        optimal_sigmas[f"{int((target_r-1)*100)}%"] = sigma_star
        target_ppl = PPL0 * target_r
        print(f"  Target: +{int((target_r-1)*100)}% PPL increase (r={target_r:.2f}, PPL={target_ppl:.4f})")
        print(f"    → σ* = {sigma_star:.6f}")
    else:
        print(f"  Warning: a={a:.4f} <= 0, cannot compute σ*")

# Select a target (e.g., 10% increase) for verification
target_r_verify = 1.10
if a > 0:
    sigma_verify = np.sqrt(np.log(target_r_verify) / a)
    
    print(f"\n" + "="*60)
    print(f"Verification: Testing σ* = {sigma_verify:.6f} for 10% PPL increase")
    print("="*60)
    
    # Inject noise with detailed recording
    original_weights_verify, weight_changes = inject_weight_noise(
        model_baseline, sigma_verify, record_changes=True
    )
    
    # Calculate perplexity
    ppl_verify = calculate_perplexity(model_baseline, tokenizer, calibration_text)
    r_verify = ppl_verify / PPL0
    
    # Restore weights after verification
    restore_weights(model_baseline, original_weights_verify)
    
    print(f"\nVerification Results:")
    print(f"  Target r: {target_r_verify:.4f}")
    print(f"  Actual r: {r_verify:.4f}")
    print(f"  Target PPL: {PPL0 * target_r_verify:.4f}")
    print(f"  Actual PPL: {ppl_verify:.4f}")
    print(f"  Error: {abs(r_verify - target_r_verify)/target_r_verify * 100:.2f}%")
    
    # Print detailed weight changes
    print(f"\n" + "="*60)
    print(f"Detailed Weight Modifications (σ = {sigma_verify:.6f})")
    print("="*60)
    print(f"\nTotal layers modified: {len(weight_changes)}")
    print(f"\nShowing first 10 layers:\n")
    
    for i, change in enumerate(weight_changes[:10]):
        print(f"Layer {i+1}: {change['layer']}")
        print(f"  Shape: {change['shape']}, Total params: {change['num_params']:,}")
        print(f"  Original weights (first 5 values):")
        print(f"    {change['original_values_sample']}")
        print(f"  Noise added (first 5 values):")
        print(f"    {change['noise_values_sample']}")
        print(f"  Noisy weights (first 5 values):")
        print(f"    {change['noisy_values_sample']}")
        print(f"  Statistics:")
        print(f"    Original mean: {change['original_mean']:.6f}, std: {change['original_std']:.6f}")
        print(f"    Noise mean: {change['noise_mean']:.6f}, std: {change['noise_std']:.6f}")
        print()
    
    # Save detailed results
    detailed_results = {
        'baseline_ppl': PPL0,
        'calibration_prompts': len(calibration_prompts),
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
        'weight_changes_sample': weight_changes[:20],  # Save first 20 layers
        'timestamp': datetime.now().isoformat()
    }
    
    with open("weight_noise_injection_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=4)
    
    print(f"Detailed results saved to: weight_noise_injection_results.json")
    
    # Create summary report
    with open("weight_noise_injection_summary.txt", 'w', encoding='utf-8') as f:
        f.write("Weight Noise Injection Analysis Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Baseline PPL0: {PPL0:.4f}\n")
        f.write(f"Analysis set: {len(calibration_prompts)} prompts (all prompts)\n\n")
        
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
        f.write(f"  Error: {abs(r_verify - target_r_verify)/target_r_verify * 100:.2f}%\n\n")
        
        f.write(f"Total layers modified: {len(weight_changes)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Summary saved to: weight_noise_injection_summary.txt")
    
    # Try to plot results
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: PPL vs sigma
        plt.subplot(1, 2, 1)
        sigmas = [res['sigma'] for res in results]
        ppls = [res['ppl'] for res in results]
        plt.plot(sigmas, ppls, 'o-', label='Measured PPL')
        plt.axhline(y=PPL0, color='r', linestyle='--', label=f'Baseline PPL0={PPL0:.4f}')
        plt.axhline(y=ppl_verify, color='g', linestyle='--', label=f'Verified PPL={ppl_verify:.4f}')
        plt.axvline(x=sigma_verify, color='g', linestyle=':', alpha=0.5)
        plt.xlabel('σ (noise std)')
        plt.ylabel('Perplexity')
        plt.title('Perplexity vs Noise Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Plot 2: log(r) vs sigma^2 (verify linear relationship)
        plt.subplot(1, 2, 2)
        plt.plot(sigma_squared, log_r, 'o', label='Measured log(r)')
        
        # Plot fitted line
        sigma_sq_range = np.linspace(0, max(sigma_squared)*1.1, 100)
        log_r_fitted = a * sigma_sq_range + b
        plt.plot(sigma_sq_range, log_r_fitted, 'r-', label=f'Fit: log(r) = {a:.4f}σ² + {b:.4f}')
        
        plt.xlabel('σ² (noise variance)')
        plt.ylabel('log(r)')
        plt.title('Quadratic Model Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('weight_noise_injection_plot.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: weight_noise_injection_plot.png")
        
    except Exception as e:
        print(f"Note: Plotting skipped (matplotlib not available): {e}")

print("\n" + "="*60)
print("Weight Noise Injection Analysis Complete!")
print("="*60)
