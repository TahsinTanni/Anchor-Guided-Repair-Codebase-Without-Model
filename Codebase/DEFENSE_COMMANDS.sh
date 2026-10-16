#!/bin/bash
#
# Quick Reference: Running the Realistic Defense
#
# This script shows example commands for running the attacked-model-anchored defense.
# You can copy-paste these commands directly into your terminal.

# ==============================================================================
# SINGLE MODEL DEFENSE EXAMPLES
# ==============================================================================

# Example 1: Defend 10% attacked model (standard hyperparameters)
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct \
    --lambda_anchor 2e-4 \
    --max_steps 60 \
    --lr 2e-5 \
    --batch_size 1 \
    --grad_accum_steps 2 \
    --fp16

# Example 2: Defend 20% attacked model (may need more aggressive repair)
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_20pct \
    --output_dir saved_models/defended_realistic_20pct \
    --lambda_anchor 1e-4 \
    --max_steps 80 \
    --lr 2e-5 \
    --batch_size 1 \
    --grad_accum_steps 2 \
    --fp16

# Example 3: Defend 5% attacked model (may need less aggressive repair)
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_5pct \
    --output_dir saved_models/defended_realistic_5pct \
    --lambda_anchor 3e-4 \
    --max_steps 40 \
    --lr 2e-5 \
    --batch_size 1 \
    --grad_accum_steps 2 \
    --fp16

# ==============================================================================
# BATCH DEFENSE (ALL ATTACK LEVELS)
# ==============================================================================

# Run defense for all attack levels (5%, 10%, 15%, 20%)
python run_realistic_defense.py

# ==============================================================================
# EVALUATION EXAMPLES
# ==============================================================================

# Evaluate defended model (single)
python evaluate_defended_model.py \
    --defended_path saved_models/defended_realistic_10pct

# Evaluate all defended models
for pct in 5pct 10pct 15pct 20pct; do
    python evaluate_defended_model.py \
        --defended_path saved_models/defended_realistic_${pct}
done

# ==============================================================================
# HYPERPARAMETER TUNING EXAMPLES
# ==============================================================================

# If defended PPL is too close to attacked PPL (under-recovery):
# - Decrease lambda_anchor (e.g., 1e-4 instead of 2e-4)
# - Increase max_steps (e.g., 80-100 instead of 60)
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct_more_aggressive \
    --lambda_anchor 1e-4 \
    --max_steps 100 \
    --lr 2e-5 \
    --batch_size 1 \
    --grad_accum_steps 2 \
    --fp16

# If defended PPL beats clean PPL (over-recovery):
# - Increase lambda_anchor (e.g., 3e-4 instead of 2e-4)
# - Decrease max_steps (e.g., 40 instead of 60)
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct_less_aggressive \
    --lambda_anchor 3e-4 \
    --max_steps 40 \
    --lr 1e-5 \
    --batch_size 1 \
    --grad_accum_steps 2 \
    --fp16

# ==============================================================================
# OOM TROUBLESHOOTING
# ==============================================================================

# If you encounter OOM errors, reduce sequence length:
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct \
    --lambda_anchor 2e-4 \
    --max_steps 60 \
    --seq_len 1024 \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --fp16

# ==============================================================================
# KEY DIFFERENCES FROM OLD VERSION
# ==============================================================================

# OLD VERSION (Unrealistic - anchored to clean model):
# python defense_anchor_finetune.py \
#     --clean_model_path saved_models/wikitext2_clean_fp16 \     ← Required clean model
#     --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
#     --output_dir saved_models/defended_noisy_fp16_10pct \
#     --lambda_anchor 1e-3 \
#     --max_steps 1000

# NEW VERSION (Realistic - anchored to attacked model):
# python defense_anchor_finetune.py \
#     --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \  ← Only attacked model
#     --output_dir saved_models/defended_realistic_10pct \
#     --lambda_anchor 2e-4 \                                            ← Lower lambda
#     --max_steps 60                                                    ← Much fewer steps

# ==============================================================================
# EXPECTED RESULTS
# ==============================================================================

# For 10% attack:
# PPL_clean    ≈ 12.50 (original model, not used for anchoring)
# PPL_attacked ≈ 25.00 (downloaded vulnerable model)
# PPL_defended ≈ 17.50 (after defense, 60% recovery)
#
# Target inequality: PPL_clean < PPL_defended < PPL_attacked
# Recovery rate: 60-80% improvement over attacked model
