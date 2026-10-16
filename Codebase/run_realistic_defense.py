"""
Example script demonstrating the realistic attacked-model-anchored defense.

This script shows how to run the defense for different attack levels (5%, 10%, 15%, 20%)
using the updated attacked-model-anchored methodology.

REALISTIC SCENARIO:
- User downloads an attacked model (e.g., wikitext2_noisy_fp16_10pct)
- User has NO access to clean weights
- Defense repairs the attacked model by anchoring to its own weights
- Goal: Partial recovery (PPL_clean < PPL_defended < PPL_attacked)
"""

import os
import subprocess
import sys

# Configuration
ATTACKED_MODELS = [
    ("saved_models/wikitext2_noisy_fp16_5pct", "5pct"),
    ("saved_models/wikitext2_noisy_fp16_10pct", "10pct"),
    ("saved_models/wikitext2_noisy_fp16_15pct", "15pct"),
    ("saved_models/wikitext2_noisy_fp16_20pct", "20pct"),
]

# Defense hyperparameters (tuned for partial recovery)
LAMBDA_ANCHOR = 2e-4  # Anchor penalty weight
MAX_STEPS = 60        # Short training to avoid over-recovery
LEARNING_RATE = 2e-5  # Conservative LR
BATCH_SIZE = 1        # OOM safety
GRAD_ACCUM = 2        # Effective batch size = 2

def run_defense(attacked_model_path, output_suffix):
    """Run defense for a specific attacked model."""
    
    output_dir = f"saved_models/defended_realistic_{output_suffix}"
    
    print("\n" + "="*80)
    print(f"Running Realistic Defense for {output_suffix} Attack")
    print("="*80)
    print(f"Attacked model: {attacked_model_path}")
    print(f"Output: {output_dir}")
    print(f"Anchor reference: ATTACKED MODEL (not clean)")
    print("="*80 + "\n")
    
    cmd = [
        "python", "defense_anchor_finetune.py",
        "--attacked_model_path", attacked_model_path,
        "--output_dir", output_dir,
        "--lambda_anchor", str(LAMBDA_ANCHOR),
        "--max_steps", str(MAX_STEPS),
        "--lr", str(LEARNING_RATE),
        "--batch_size", str(BATCH_SIZE),
        "--grad_accum_steps", str(GRAD_ACCUM),
        "--fp16"
    ]
    
    # Run defense
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✓ Defense complete for {output_suffix}")
        print(f"  Defended model saved to: {output_dir}")
    else:
        print(f"\n✗ Defense failed for {output_suffix}")
        sys.exit(1)
    
    return output_dir

def main():
    print("\n" + "="*80)
    print("REALISTIC ATTACKED-MODEL-ANCHORED DEFENSE")
    print("="*80)
    print("\nMethodology:")
    print("  - Anchor reference: ATTACKED model weights (not clean)")
    print("  - Goal: Partial recovery (PPL_clean < PPL_defended < PPL_attacked)")
    print("  - No access to clean weights (realistic scenario)")
    print("\nHyperparameters:")
    print(f"  - Lambda anchor: {LAMBDA_ANCHOR}")
    print(f"  - Max steps: {MAX_STEPS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRAD_ACCUM}")
    print("="*80 + "\n")
    
    defended_models = []
    
    # Run defense for each attack level
    for attacked_path, suffix in ATTACKED_MODELS:
        if not os.path.exists(attacked_path):
            print(f"⚠ Warning: Attacked model not found: {attacked_path}")
            print(f"  Skipping {suffix} defense...")
            continue
        
        defended_dir = run_defense(attacked_path, suffix)
        defended_models.append((defended_dir, suffix))
    
    # Summary
    print("\n" + "="*80)
    print("DEFENSE SUMMARY")
    print("="*80)
    print(f"\nDefended models created: {len(defended_models)}")
    for defended_dir, suffix in defended_models:
        print(f"  - {suffix}: {defended_dir}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Evaluate defended models:")
    print("   python evaluate_defended_model.py --defended_path <defended_dir>")
    print("\n2. Verify partial recovery:")
    print("   - PPL_defended should be better than PPL_attacked")
    print("   - PPL_defended should be worse than PPL_clean")
    print("   - Target recovery: 60-80% improvement")
    print("\n3. Plot results:")
    print("   python plot_defense_comparison.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
