"""
Generate Attacked Models FROM Task-Adapted Clean Baseline

PURPOSE:
--------
This script generates attacked models by injecting weight noise into the
TASK-ADAPTED clean baseline (not the original zero-shot clean model).

WORKFLOW (METHOD A):
--------------------
1. Load task-adapted clean baseline from: saved_models/wikitext2_clean_adapted_fp16
2. Inject Gaussian noise at specified noise levels (5%, 10%, 15%, 20%)
3. Save attacked models to:
   - saved_models/wikitext2_attacked_adapted_fp16_5pct
   - saved_models/wikitext2_attacked_adapted_fp16_10pct
   - saved_models/wikitext2_attacked_adapted_fp16_15pct
   - saved_models/wikitext2_attacked_adapted_fp16_20pct

IMPORTANT DIFFERENCE FROM weight_noise_injection.py:
-----------------------------------------------------
- OLD: Inject noise into original zero-shot clean Qwen-0.5B
- NEW: Inject noise into task-adapted clean baseline
- This ensures attacked models start from a realistic baseline

USAGE:
------
python weight_noise_from_adapted.py \\
    --base_model_path saved_models/wikitext2_clean_adapted_fp16 \\
    --noise_levels 5 10 15 20 \\
    --output_dir saved_models

EXPECTED OUTPUT:
----------------
- Attacked model checkpoints at each noise level
- Perplexity measurements for each attacked model
- JSON log with attack details: attacked_models_from_adapted_log.json

NEXT STEPS:
-----------
1. Defend attacked models using defense_anchor_finetune_adapted.py
2. Evaluate recovery with proper baseline comparison
"""

import argparse
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime


def calculate_perplexity(model, tokenizer, text, desc=""):
    """Calculate perplexity on text for evaluation."""
    device = model.device
    
    print(f"\nCalculating perplexity for {desc}...")
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = 2048
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
    perplexity_value = ppl.item()
    
    print(f"  Perplexity: {perplexity_value:.2f}")
    
    return perplexity_value


def inject_weight_noise(model, noise_pct):
    """
    Inject Gaussian noise into model weights IN-PLACE.
    
    Args:
        model: PyTorch model
        noise_pct: Target noise level as percentage (e.g., 10 for 10%)
    
    Uses empirically calibrated sigma values for realistic PPL degradation.
    Based on typical Qwen-0.5B behavior where small sigma (~0.001-0.005) 
    produces 10-20% PPL increases.
    """
    # Empirically calibrated sigma values for Qwen-0.5B
    # These values produce realistic PPL increases of 5-20%
    sigma_map = {
        5: 0.0010,   # ~5-10% PPL increase
        10: 0.0015,  # ~10-15% PPL increase
        15: 0.0020,  # ~15-20% PPL increase
        20: 0.0025   # ~20-25% PPL increase
    }
    
    sigma = sigma_map.get(noise_pct, 0.0015)  # Default to 10% level
    
    print(f"\nInjecting noise for {noise_pct}% target PPL increase:")
    print(f"  Using calibrated σ: {sigma:.6f}")
    
    noise_stats = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.device.type != 'meta':
            # Generate Gaussian noise with mean=0, std=sigma
            noise = torch.randn_like(param) * sigma
            
            # Record statistics
            noise_stats.append({
                'layer': name,
                'shape': list(param.shape),
                'num_params': param.numel(),
                'original_mean': param.data.mean().item(),
                'original_std': param.data.std().item(),
                'noise_mean': noise.mean().item(),
                'noise_std': noise.std().item()
            })
            
            # Add noise IN-PLACE
            param.data.add_(noise)
    
    print(f"  Noise injected into {len(noise_stats)} parameter tensors")
    
    return sigma, noise_stats


def main():
    parser = argparse.ArgumentParser(description="Generate Attacked Models from Task-Adapted Clean Baseline")
    parser.add_argument("--base_model_path", type=str, 
                        default="saved_models/wikitext2_clean_adapted_fp16",
                        help="Path to task-adapted clean baseline")
    parser.add_argument("--noise_levels", type=int, nargs='+', default=[5, 10, 15, 20],
                        help="Noise levels as percentages (e.g., 5 10 15 20)")
    parser.add_argument("--output_dir", type=str, default="saved_models",
                        help="Output directory for attacked models")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset name for perplexity evaluation")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset configuration")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Generate Attacked Models from Task-Adapted Clean Baseline")
    print("="*70)
    print(f"Base model: {args.base_model_path}")
    print(f"Noise levels: {args.noise_levels}%")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    # Load test dataset
    print("\nLoading test dataset...")
    from datasets import load_dataset
    dataset = load_dataset(args.dataset, args.dataset_config)
    test_text = "\n\n".join([text for text in dataset["test"]["text"] if text.strip()])
    
    # Measure baseline (task-adapted clean) perplexity
    print("\n" + "="*70)
    print("Loading Task-Adapted Clean Baseline")
    print("="*70)
    
    model_clean = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager',
        _attn_implementation_internal='eager'
    )
    
    print(f"Model loaded on {model_clean.device}")
    
    baseline_ppl = calculate_perplexity(model_clean, tokenizer, test_text, 
                                         desc="Task-Adapted Clean Baseline")
    
    # Store results
    results = {
        "base_model_path": args.base_model_path,
        "baseline_ppl": baseline_ppl,
        "noise_levels": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Generate attacked models at each noise level
    for noise_pct in args.noise_levels:
        print("\n" + "="*70)
        print(f"Generating Attacked Model: {noise_pct}% Noise")
        print("="*70)
        
        # Reload clean model for each noise level
        model_attacked = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation='eager',
            _attn_implementation_internal='eager'
        )
        
        # Inject noise
        sigma, noise_stats = inject_weight_noise(model_attacked, noise_pct)
        
        # Measure perplexity
        attacked_ppl = calculate_perplexity(model_attacked, tokenizer, test_text,
                                            desc=f"Attacked ({noise_pct}% noise)")
        
        # Calculate actual increase
        actual_increase_pct = ((attacked_ppl - baseline_ppl) / baseline_ppl) * 100
        
        print(f"\n  Baseline PPL: {baseline_ppl:.2f}")
        print(f"  Attacked PPL: {attacked_ppl:.2f}")
        print(f"  Target increase: {noise_pct}%")
        print(f"  Actual increase: {actual_increase_pct:.1f}%")
        
        # Save attacked model
        output_path = os.path.join(args.output_dir, 
                                   f"wikitext2_attacked_adapted_fp16_{noise_pct}pct")
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\nSaving attacked model to: {output_path}")
        model_attacked.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Record results
        results["noise_levels"].append({
            "noise_pct": noise_pct,
            "sigma": sigma,
            "baseline_ppl": baseline_ppl,
            "attacked_ppl": attacked_ppl,
            "target_increase_pct": noise_pct,
            "actual_increase_pct": actual_increase_pct,
            "output_path": output_path,
            "noise_stats_sample": noise_stats[:3]  # First 3 layers
        })
        
        # Clean up
        del model_attacked
        torch.cuda.empty_cache()
    
    # Save results log
    log_file = "attacked_models_from_adapted_log.json"
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("ATTACKED MODEL GENERATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {log_file}")
    print("\nGenerated models:")
    for level_result in results["noise_levels"]:
        print(f"  - {level_result['output_path']}")
        print(f"    PPL: {level_result['attacked_ppl']:.2f} (+{level_result['actual_increase_pct']:.1f}%)")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("Defend attacked models using anchor-based fine-tuning:")
    for noise_pct in args.noise_levels:
        attacked_path = os.path.join(args.output_dir, 
                                     f"wikitext2_attacked_adapted_fp16_{noise_pct}pct")
        defended_path = os.path.join(args.output_dir, 
                                     f"wikitext2_defended_adapted_fp16_{noise_pct}pct")
        print(f"\npython defense_anchor_finetune_adapted.py \\")
        print(f"    --clean_model_path {args.base_model_path} \\")
        print(f"    --attacked_model_path {attacked_path} \\")
        print(f"    --output_dir {defended_path} \\")
        print(f"    --lambda_anchor 1e-4 \\")
        print(f"    --max_steps 100")
    print("="*70)


if __name__ == "__main__":
    main()
