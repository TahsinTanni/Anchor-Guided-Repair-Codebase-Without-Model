"""
Run defense and evaluation for all noise levels (5%, 10%, 15%, 20%)
Uses the same hyperparameters for all levels: lambda_anchor=1e-5, max_steps=250
"""
import subprocess
import os

# Configuration
NOISE_LEVELS = [5, 10, 15, 20]
LAMBDA_ANCHOR = "1e-5"
MAX_STEPS = "250"
LR = "2e-5"  # Use --lr instead of --learning_rate
GRAD_ACCUM = "2"  # Use --grad_accum_steps instead of --gradient_accumulation_steps
PYTHON_EXE = r"C:/Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection/.venv/Scripts/python.exe"

CLEAN_MODEL = "saved_models/wikitext2_clean_adapted_fp16"

print("="*80)
print("Running Defense and Evaluation for All Noise Levels")
print("="*80)
print(f"Parameters: lambda_anchor={LAMBDA_ANCHOR}, max_steps={MAX_STEPS}")
print(f"Noise levels: {NOISE_LEVELS}%")
print("="*80)

for noise in NOISE_LEVELS:
    attacked_path = f"saved_models/wikitext2_attacked_adapted_fp16_{noise}pct"
    defended_path = f"saved_models/wikitext2_defended_adapted_fp16_{noise}pct"
    results_file = f"defended_adapted_{noise}pct_results.json"
    
    # Check if defended model already exists
    if os.path.exists(defended_path):
        print(f"\n✓ Defended model for {noise}% already exists, skipping defense training")
    else:
        print(f"\n{'='*80}")
        print(f"STEP 1: Defending {noise}% Attacked Model")
        print(f"{'='*80}")
        
        defense_cmd = [
            PYTHON_EXE,
            "defense_anchor_finetune_adapted.py",
            "--clean_model_path", CLEAN_MODEL,
            "--attacked_model_path", attacked_path,
            "--output_dir", defended_path,
            "--lambda_anchor", LAMBDA_ANCHOR,
            "--max_steps", MAX_STEPS,
            "--lr", LR,
            "--grad_accum_steps", GRAD_ACCUM
        ]
        
        result = subprocess.run(defense_cmd, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"✗ Defense training failed for {noise}%")
            continue
        print(f"✓ Defense complete for {noise}%")
    
    # Check if results file already exists
    if os.path.exists(results_file):
        print(f"\n✓ Evaluation results for {noise}% already exist, skipping evaluation")
    else:
        print(f"\n{'='*80}")
        print(f"STEP 2: Evaluating {noise}% Defended Model")
        print(f"{'='*80}")
        
        eval_cmd = [
            PYTHON_EXE,
            "evaluate_defended_adapted.py",
            "--clean_model_path", CLEAN_MODEL,
            "--attacked_model_path", attacked_path,
            "--defended_model_path", defended_path,
            "--output_file", results_file
        ]
        
        result = subprocess.run(eval_cmd, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"✗ Evaluation failed for {noise}%")
            continue
        print(f"✓ Evaluation complete for {noise}%")

print(f"\n{'='*80}")
print("STEP 3: Generating Plots")
print(f"{'='*80}")

plot_cmd = [PYTHON_EXE, "plot_method_a_results.py"]
result = subprocess.run(plot_cmd, cwd=os.getcwd())
if result.returncode == 0:
    print("✓ All plots generated successfully!")
else:
    print("✗ Plot generation failed")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
print("Results saved to:")
for noise in NOISE_LEVELS:
    results_file = f"defended_adapted_{noise}pct_results.json"
    if os.path.exists(results_file):
        print(f"  ✓ {results_file}")
    else:
        print(f"  ✗ {results_file} (missing)")
print("\nPlots generated:")
for noise in NOISE_LEVELS:
    plot_file = f"method_a_defense_results_{noise}pct.png"
    if os.path.exists(plot_file):
        print(f"  ✓ {plot_file}")
    else:
        print(f"  ✗ {plot_file} (missing)")
