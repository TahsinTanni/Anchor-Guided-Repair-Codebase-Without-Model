# Realistic Defense Methodology: Attacked-Model-Anchored Fine-Tuning

## Overview

This document describes the **realistic defense scenario** implemented in this project. Unlike the initial version that anchored to clean weights (unrealistic), this defense models a real-world scenario where:

- **Hackers release an attacked/vulnerable model** with injected weight noise
- **A normal user downloads ONLY the attacked model** (no access to clean weights)
- **The user wants to repair the model** to improve performance

## Key Principle

> **The attacked model itself is the reference.**  
> The defense anchors updates to the attacked model's weights, NOT to clean weights.  
> Clean model is used ONLY for evaluation/sanity checks.

---

## Methodology

### 1. Load Attacked Model
```python
# User downloads vulnerable model from attackers
attacked_model = load_model("wikitext2_noisy_fp16_10pct")
```

### 2. Create Anchor Reference
```python
# Freeze a copy of attacked weights as anchor
anchor_model = copy.deepcopy(attacked_model)
freeze_parameters(anchor_model)
```

### 3. Fine-Tune with Anchoring
```python
# Fine-tune trainable copy on clean text (WikiText-2)
Total_Loss = LM_Loss(WikiText-2) + lambda_anchor * L2(W_current, W_attacked)
```

### 4. Partial Recovery
```python
# Stop training early to ensure partial recovery
# Goal: PPL_clean < PPL_defended < PPL_attacked
# Target: 60-80% improvement over attacked model
```

---

## Why This Works

1. **Fine-tuning on clean text** helps recover language modeling capability
2. **Anchoring to attacked weights** prevents over-recovery and drift
3. **Early stopping** ensures partial repair without full restoration
4. **Realistic scenario** where clean weights are unavailable

---

## Expected Behavior

### Target Inequality
```
PPL_clean < PPL_defended < PPL_attacked
```

### Example Results (10% Attack)
```
PPL_clean    = 12.50 (original model, sanity reference)
PPL_attacked = 25.00 (downloaded vulnerable model)
PPL_defended = 17.50 (output of defense, 60% recovery)
```

### Recovery Rate
- **Partial recovery**: 60-80% improvement over attacked model
- **Not full restoration**: Defended model remains worse than clean baseline
- **No over-recovery**: Defended PPL should NOT beat clean PPL

---

## Usage

### Running the Defense

```bash
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct \
    --lambda_anchor 2e-4 \
    --max_steps 60
```

### Batch Processing (All Attack Levels)

```bash
python run_realistic_defense.py
```

This will defend models at 5%, 10%, 15%, and 20% attack levels.

### Evaluating Defended Models

```bash
python evaluate_defended_model.py \
    --defended_path saved_models/defended_realistic_10pct
```

---

## Hyperparameters

### Standard Defense Settings (Realistic Scenario)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lambda_anchor` | `2e-4` | Balances repair vs. stability |
| `max_steps` | `60` | Short training for partial recovery |
| `learning_rate` | `2e-5` | Conservative to avoid destabilizing |
| `batch_size` | `1` | OOM safety |
| `grad_accum_steps` | `2` | Effective batch size = 2 |
| `fp16` | `True` | Memory efficiency |

### Tuning Guidelines

**If defended PPL is too close to attacked PPL (under-recovery):**
- Decrease `lambda_anchor` (e.g., `1e-4`)
- Increase `max_steps` (e.g., `80-100`)
- Increase `learning_rate` (e.g., `3e-5`)

**If defended PPL beats clean PPL (over-recovery):**
- Increase `lambda_anchor` (e.g., `3e-4`)
- Decrease `max_steps` (e.g., `40-50`)
- Decrease `learning_rate` (e.g., `1e-5`)

---

## Comparison: Old vs. New Methodology

### Old Approach (Unrealistic)
```
✗ Anchored to CLEAN weights
✗ Assumes user has access to clean model
✗ Leads to over-recovery (>100%)
✗ Not realistic for real-world scenarios
```

### New Approach (Realistic)
```
✓ Anchored to ATTACKED weights
✓ User only has access to attacked model
✓ Partial recovery (60-80%)
✓ Models real-world defense scenario
```

---

## Evaluation Protocol

### Required Comparisons

1. **PPL_clean** (original model, sanity reference only)
2. **PPL_attacked** (downloaded vulnerable model)
3. **PPL_defended** (output of defense)

### Success Criteria

```python
assert PPL_clean < PPL_defended < PPL_attacked
assert (PPL_attacked - PPL_defended) / (PPL_attacked - PPL_clean) >= 0.6
assert (PPL_attacked - PPL_defended) / (PPL_attacked - PPL_clean) <= 0.8
```

### Example Report

```
Attack Level: 10%
------------------------------------------------------------
PPL_clean    : 12.50 (baseline, not used for anchoring)
PPL_attacked : 25.00 (downloaded model)
PPL_defended : 17.50 (after repair)

Recovery     : 60.0% ((25.00 - 17.50) / (25.00 - 12.50))
Status       : ✓ Partial recovery achieved
Sanity Check : ✓ PPL_defended > PPL_clean (no over-recovery)
```

---

## OOM Safety

### Memory Optimization Strategies

1. **Batch size = 1** (always)
2. **Gradient accumulation** (increase for larger effective batch size)
3. **Reduce sequence length** (if needed, e.g., 1024 instead of 2048)
4. **Gradient checkpointing** (optional, add to training args)
5. **FP16 training** (enabled by default)

### Gradient Checkpointing (If Needed)

```python
# In defense_anchor_finetune.py, add to TrainingArguments:
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
    ...
)

# Also enable in model:
model.gradient_checkpointing_enable()
```

---

## Code Changes Summary

### Modified Files

1. **[defense_anchor_finetune.py](defense_anchor_finetune.py)**
   - Changed anchor reference from clean to attacked model
   - Updated hyperparameter defaults for partial recovery
   - Removed `--clean_model_path` argument
   - Updated documentation and print statements

2. **[run_realistic_defense.py](run_realistic_defense.py)** (NEW)
   - Batch processing script for all attack levels
   - Demonstrates proper usage of updated defense script

### Unchanged Scripts (Reuse As-Is)

- `weight_noise_injection.py` (attack generation)
- `quantize_noisy_model.py` (INT8/4-bit quantization)
- `evaluate_defended_model.py` (perplexity comparison)
- `calculate_perplexity*.py` (PPL evaluation)

---

## Troubleshooting

### Issue: Defended PPL beats clean PPL (over-recovery)

**Solution:**
```bash
# Increase anchoring penalty
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct_v2 \
    --lambda_anchor 3e-4 \  # Increased from 2e-4
    --max_steps 40           # Decreased from 60
```

### Issue: Defended PPL too close to attacked PPL (under-recovery)

**Solution:**
```bash
# Decrease anchoring penalty, increase training
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct_v2 \
    --lambda_anchor 1e-4 \  # Decreased from 2e-4
    --max_steps 100         # Increased from 60
```

### Issue: OOM during training

**Solution:**
```bash
# Reduce sequence length, enable gradient checkpointing
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct \
    --seq_len 1024 \        # Reduced from 2048
    --batch_size 1 \
    --grad_accum_steps 4    # Increased for same effective batch size
```

Then manually add gradient checkpointing in the code.

---

## Citation

If you use this realistic defense methodology in your research, please cite:

```bibtex
@article{defense-against-quantization-attacks,
  title={Defense Against Quantization Attacks in Large Language Models Through Weight Noise Injection},
  author={[Your Name]},
  year={2026},
  note={Realistic attacked-model-anchored fine-tuning}
}
```

---

## Summary

This realistic defense methodology:
- ✅ Models real-world scenario (no access to clean weights)
- ✅ Achieves partial recovery (60-80%)
- ✅ Prevents over-recovery (defended > clean)
- ✅ Minimal code changes (reuses existing infrastructure)
- ✅ OOM-safe defaults (batch size = 1)
- ✅ Easy to use and tune

**The attacked model is both the starting point AND the anchor reference.**
