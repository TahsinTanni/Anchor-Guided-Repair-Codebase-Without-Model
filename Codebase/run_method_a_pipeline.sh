#!/bin/bash
# Method A: Complete Pipeline Script
# Run this script to execute the full task-adapted baseline methodology

# Set strict error handling
set -e

echo "============================================================================"
echo "Method A: Task-Adapted Clean Baseline Pipeline"
echo "============================================================================"
echo ""

# Configuration
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
CLEAN_ADAPTED_PATH="saved_models/wikitext2_clean_adapted_fp16"
NOISE_LEVELS="5 10 15 20"
LAMBDA_ANCHOR="1e-4"
DEFENSE_STEPS="100"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Clean adapted path: $CLEAN_ADAPTED_PATH"
echo "  Noise levels: $NOISE_LEVELS%"
echo "  Lambda anchor: $LAMBDA_ANCHOR"
echo "  Defense steps: $DEFENSE_STEPS"
echo ""

# ============================================================================
# STEP 1: Train Task-Adapted Clean Baseline (ONE-TIME)
# ============================================================================
echo "============================================================================"
echo "STEP 1: Training Task-Adapted Clean Baseline"
echo "============================================================================"
echo ""

if [ -d "$CLEAN_ADAPTED_PATH" ]; then
    echo "Task-adapted clean baseline already exists at: $CLEAN_ADAPTED_PATH"
    echo "Skipping Step 1. Delete the directory to retrain."
else
    python train_task_adapted_clean.py \
        --model_name "$MODEL_NAME" \
        --output_dir "$CLEAN_ADAPTED_PATH" \
        --max_steps 200 \
        --learning_rate 2e-5 \
        --gradient_accumulation_steps 4 \
        --fp16
    
    echo ""
    echo "Task-adapted clean baseline training complete!"
fi

echo ""

# ============================================================================
# STEP 2: Generate Attacked Models from Task-Adapted Baseline
# ============================================================================
echo "============================================================================"
echo "STEP 2: Generating Attacked Models"
echo "============================================================================"
echo ""

python weight_noise_from_adapted.py \
    --base_model_path "$CLEAN_ADAPTED_PATH" \
    --noise_levels $NOISE_LEVELS \
    --output_dir saved_models

echo ""
echo "Attacked model generation complete!"
echo ""

# ============================================================================
# STEP 3: Defend Attacked Models (Anchor to Task-Adapted Clean)
# ============================================================================
echo "============================================================================"
echo "STEP 3: Defending Attacked Models"
echo "============================================================================"
echo ""

for NOISE in $NOISE_LEVELS; do
    ATTACKED_PATH="saved_models/wikitext2_attacked_adapted_fp16_${NOISE}pct"
    DEFENDED_PATH="saved_models/wikitext2_defended_adapted_fp16_${NOISE}pct"
    
    echo "Defending attacked model with ${NOISE}% noise..."
    echo "  Attacked: $ATTACKED_PATH"
    echo "  Output: $DEFENDED_PATH"
    echo ""
    
    python defense_anchor_finetune_adapted.py \
        --clean_model_path "$CLEAN_ADAPTED_PATH" \
        --attacked_model_path "$ATTACKED_PATH" \
        --output_dir "$DEFENDED_PATH" \
        --lambda_anchor $LAMBDA_ANCHOR \
        --max_steps $DEFENSE_STEPS \
        --learning_rate 2e-5 \
        --gradient_accumulation_steps 2
    
    echo ""
    echo "Defense complete for ${NOISE}% noise level!"
    echo ""
done

echo ""
echo "All defenses complete!"
echo ""

# ============================================================================
# STEP 4: Evaluate Defended Models
# ============================================================================
echo "============================================================================"
echo "STEP 4: Evaluating Defended Models"
echo "============================================================================"
echo ""

for NOISE in $NOISE_LEVELS; do
    ATTACKED_PATH="saved_models/wikitext2_attacked_adapted_fp16_${NOISE}pct"
    DEFENDED_PATH="saved_models/wikitext2_defended_adapted_fp16_${NOISE}pct"
    OUTPUT_FILE="defended_adapted_${NOISE}pct_results.json"
    
    echo "Evaluating defended model with ${NOISE}% noise..."
    echo "  Output: $OUTPUT_FILE"
    echo ""
    
    python evaluate_defended_adapted.py \
        --clean_model_path "$CLEAN_ADAPTED_PATH" \
        --attacked_model_path "$ATTACKED_PATH" \
        --defended_model_path "$DEFENDED_PATH" \
        --output_file "$OUTPUT_FILE"
    
    echo ""
    echo "Evaluation complete for ${NOISE}% noise level!"
    echo ""
done

echo ""
echo "============================================================================"
echo "PIPELINE COMPLETE"
echo "============================================================================"
echo ""
echo "Results saved to:"
for NOISE in $NOISE_LEVELS; do
    echo "  - defended_adapted_${NOISE}pct_results.json"
done
echo ""
echo "Check the JSON files for perplexity values and recovery rates."
echo "Expected: PPL_clean_adapted < PPL_defended < PPL_attacked at all levels"
echo "Expected: Recovery rates ≤ 100% (bounded and interpretable)"
echo ""
echo "============================================================================"
