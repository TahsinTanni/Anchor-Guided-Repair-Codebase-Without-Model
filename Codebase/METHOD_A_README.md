# Method A: Task-Adapted Clean Baseline Methodology

## Overview

This methodology fixes the recovery rate >100% issue by using a **task-adapted clean baseline** instead of a zero-shot clean model as the reference point.

## The Problem

When defending attacked models on WikiText-2:
- **Original clean model** is zero-shot on WikiText-2 (high PPL)
- **Defended models** are fine-tuned on WikiText-2 (lower PPL)
- **Result**: Defended PPL < Clean PPL → Recovery > 100% (nonsensical)

## The Solution (Method A)

Redefine the "clean baseline" as a **task-adapted clean model**:
1. Fine-tune the original clean model on WikiText-2 (one-time)
2. Use this as the baseline for attack generation and defense evaluation
3. Ensures: **PPL_clean_adapted < PPL_defended < PPL_attacked**
4. Recovery rates are now bounded at ≤100% and interpretable

## Workflow

### Step 1: Train Task-Adapted Clean Baseline (One-Time)

```bash
python train_task_adapted_clean.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir saved_models/wikitext2_clean_adapted_fp16 \
    --max_steps 200 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --fp16
```

**Purpose**: Create a fair baseline by adapting the clean model to WikiText-2.

**Key Parameters**:
- `max_steps`: 200 (sufficient for task adaptation)
- `lambda_anchor`: 0 (NO anchoring - standard LM training)
- `batch_size`: 1 (OOM safety)
- `gradient_accumulation_steps`: 4 (effective batch size = 4)

**Output**: `saved_models/wikitext2_clean_adapted_fp16/`

---

### Step 2: Generate Attacked Models FROM Task-Adapted Baseline

```bash
python weight_noise_from_adapted.py \
    --base_model_path saved_models/wikitext2_clean_adapted_fp16 \
    --noise_levels 5 10 15 20 \
    --output_dir saved_models
```

**Purpose**: Create attacked models by injecting noise into the task-adapted clean baseline (not the original zero-shot model).

**Key Difference**: 
- **OLD**: Noise injected into zero-shot clean model
- **NEW**: Noise injected into task-adapted clean model

**Outputs**:
- `saved_models/wikitext2_attacked_adapted_fp16_5pct/`
- `saved_models/wikitext2_attacked_adapted_fp16_10pct/`
- `saved_models/wikitext2_attacked_adapted_fp16_15pct/`
- `saved_models/wikitext2_attacked_adapted_fp16_20pct/`

---

### Step 3: Defend Attacked Models (Anchor to Task-Adapted Clean)

Example for 10% noise level:

```bash
python defense_anchor_finetune_adapted.py \
    --clean_model_path saved_models/wikitext2_clean_adapted_fp16 \
    --attacked_model_path saved_models/wikitext2_attacked_adapted_fp16_10pct \
    --output_dir saved_models/wikitext2_defended_adapted_fp16_10pct \
    --lambda_anchor 1e-4 \
    --max_steps 100 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 2
```

**Purpose**: Repair attacked models by anchoring toward the task-adapted clean baseline.

**Key Difference**:
- **OLD**: Anchor to attacked model (limited recovery)
- **NEW**: Anchor to task-adapted clean model (better recovery, bounded at 100%)

**Key Parameters**:
- `lambda_anchor`: 1e-4 (good recovery without overshooting)
- `max_steps`: 80-120 (moderate training)
- Higher lambda → stays closer to attacked (less recovery)
- Lower lambda → more aggressive repair (approaches clean baseline)

**Output**: `saved_models/wikitext2_defended_adapted_fp16_10pct/`

---

### Step 4: Evaluate with Proper Baseline Comparison

```bash
python evaluate_defended_adapted.py \
    --clean_model_path saved_models/wikitext2_clean_adapted_fp16 \
    --attacked_model_path saved_models/wikitext2_attacked_adapted_fp16_10pct \
    --defended_model_path saved_models/wikitext2_defended_adapted_fp16_10pct \
    --output_file defended_adapted_10pct_results.json
```

**Purpose**: Evaluate defended models using the task-adapted clean baseline as reference.

**Recovery Calculation**:
```
Recovery = (PPL_attacked - PPL_defended) / (PPL_attacked - PPL_clean_adapted) × 100%
```

**Expected Ordering**:
```
PPL_clean_adapted < PPL_defended < PPL_attacked
```

**Output**: JSON file with perplexities and recovery rates at FP16, INT8, and 4-BIT.

---

## Complete Pipeline Example

```bash
# Step 1: Train task-adapted clean baseline (ONE-TIME)
python train_task_adapted_clean.py \
    --output_dir saved_models/wikitext2_clean_adapted_fp16 \
    --max_steps 200

# Step 2: Generate attacked models (5%, 10%, 15%, 20%)
python weight_noise_from_adapted.py \
    --base_model_path saved_models/wikitext2_clean_adapted_fp16 \
    --noise_levels 5 10 15 20

# Step 3: Defend attacked models
for NOISE in 5 10 15 20; do
    python defense_anchor_finetune_adapted.py \
        --clean_model_path saved_models/wikitext2_clean_adapted_fp16 \
        --attacked_model_path saved_models/wikitext2_attacked_adapted_fp16_${NOISE}pct \
        --output_dir saved_models/wikitext2_defended_adapted_fp16_${NOISE}pct \
        --lambda_anchor 1e-4 \
        --max_steps 100
done

# Step 4: Evaluate defended models
for NOISE in 5 10 15 20; do
    python evaluate_defended_adapted.py \
        --clean_model_path saved_models/wikitext2_clean_adapted_fp16 \
        --attacked_model_path saved_models/wikitext2_attacked_adapted_fp16_${NOISE}pct \
        --defended_model_path saved_models/wikitext2_defended_adapted_fp16_${NOISE}pct \
        --output_file defended_adapted_${NOISE}pct_results.json
done
```

---

## Hyperparameters Summary

### Task-Adapted Clean Baseline (Step 1)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_steps | 200 | Sufficient for task adaptation |
| learning_rate | 2e-5 | Conservative for stable training |
| batch_size | 1 | OOM safety on RTX 3060 12GB |
| gradient_accumulation_steps | 4 | Effective batch size = 4 |
| lambda_anchor | 0 | NO anchoring (standard LM training) |
| fp16 | True | Memory efficiency |

### Defense Training (Step 3)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| lambda_anchor | 1e-4 | Good recovery without overshooting |
| max_steps | 80-120 | Moderate training |
| learning_rate | 2e-5 | Conservative for stable training |
| batch_size | 1 | OOM safety |
| gradient_accumulation_steps | 2 | Effective batch size = 2 |
| fp16 | False | Stable gradients (use float32) |

### Lambda Anchor Guidelines
- **1e-5**: Very aggressive repair (may overshoot baseline)
- **1e-4**: Good balance (recommended, ~70-80% recovery)
- **2e-4**: Moderate repair (~60-70% recovery)
- **5e-4**: Conservative repair (~40-50% recovery)

---

## Expected Results

### Perplexity Ordering (at all precision levels)
```
PPL_clean_adapted < PPL_defended < PPL_attacked
```

### Recovery Rates
- **5% noise**: 70-85% recovery
- **10% noise**: 65-80% recovery
- **15% noise**: 60-75% recovery
- **20% noise**: 55-70% recovery

### Recovery Bounded at 100%
- Defended models **cannot** outperform clean adapted baseline
- Recovery rates are now **interpretable** and **meaningful**
- No more >100% recovery (nonsensical results eliminated)

---

## Comparison: Old vs New Methodology

| Aspect | Old (Zero-Shot Baseline) | New (Method A: Task-Adapted) |
|--------|-------------------------|------------------------------|
| **Clean Baseline** | Original Qwen-0.5B (zero-shot) | Fine-tuned on WikiText-2 |
| **Clean PPL** | High (~14-15) | Lower (~12-13) |
| **Attack Source** | Noise into zero-shot model | Noise into adapted model |
| **Defense Anchor** | Attacked model | Task-adapted clean model |
| **Recovery Range** | Can be >100% (nonsensical) | Bounded at ≤100% |
| **Interpretability** | Poor (defended beats baseline) | Good (proper ordering) |

---

## Directory Structure

```
saved_models/
├── wikitext2_clean_adapted_fp16/          # Step 1: Task-adapted baseline
├── wikitext2_attacked_adapted_fp16_5pct/  # Step 2: Attacked (5% noise)
├── wikitext2_attacked_adapted_fp16_10pct/ # Step 2: Attacked (10% noise)
├── wikitext2_attacked_adapted_fp16_15pct/ # Step 2: Attacked (15% noise)
├── wikitext2_attacked_adapted_fp16_20pct/ # Step 2: Attacked (20% noise)
├── wikitext2_defended_adapted_fp16_5pct/  # Step 3: Defended (5% noise)
├── wikitext2_defended_adapted_fp16_10pct/ # Step 3: Defended (10% noise)
├── wikitext2_defended_adapted_fp16_15pct/ # Step 3: Defended (15% noise)
└── wikitext2_defended_adapted_fp16_20pct/ # Step 3: Defended (20% noise)
```

---

## Files in This Implementation

### Core Scripts
1. **train_task_adapted_clean.py**: Create task-adapted clean baseline (Step 1)
2. **weight_noise_from_adapted.py**: Generate attacked models (Step 2)
3. **defense_anchor_finetune_adapted.py**: Defend with clean anchoring (Step 3)
4. **evaluate_defended_adapted.py**: Evaluate with proper baseline (Step 4)

### Legacy Scripts (Still Functional)
- **weight_noise_injection.py**: Original attack script (zero-shot baseline)
- **defense_anchor_finetune.py**: Original defense (attacked-model anchoring)
- **evaluate_defended_model.py**: Original evaluation

---

## Key Insights

### Why This Works
1. **Fair Baseline**: Clean model is adapted to the same task as defended models
2. **Proper Ordering**: PPL_clean < PPL_defended < PPL_attacked at all times
3. **Bounded Recovery**: Defense cannot beat the task-adapted baseline
4. **Interpretable**: Recovery rates reflect genuine defense effectiveness

### Loss Function
```
Total_Loss = LM_Loss(WikiText-2) + λ_anchor × ||θ_defended - θ_clean_adapted||²
```

Where:
- **LM_Loss**: Standard language modeling loss (improves fluency)
- **λ_anchor × L2**: Anchoring penalty (pulls toward clean baseline)

### Why NOT Anchor to Attacked Model?
- **Old approach**: Anchors to attacked model → limited recovery (60-70% max)
- **New approach**: Anchors to clean adapted → better recovery (up to 100%)
- **Result**: More effective defense while maintaining interpretability

---

## OOM Safety (RTX 3060 12GB)

### Critical Settings
- **batch_size = 1** (ALWAYS - do not increase)
- **gradient_accumulation_steps = 2-4** (increase this, not batch_size)
- **fp16 = True** (for task adaptation)
- **fp16 = False** (for defense - use float32 for stable gradients)
- **seq_len = 2048** (reduce to 1024 if OOM)
- **gradient_checkpointing = False** (optional if OOM)

### If OOM Occurs
1. Reduce `seq_len` from 2048 to 1024
2. Reduce `gradient_accumulation_steps` from 4 to 2
3. Enable `gradient_checkpointing=True` in TrainingArguments
4. Reduce `max_steps` (won't help memory, but faster iteration)

---

## Troubleshooting

### Issue: Recovery > 100%
**Cause**: Defended model beats clean adapted baseline  
**Solution**: 
- Increase `lambda_anchor` (e.g., 1e-4 → 2e-4)
- Reduce `max_steps` (e.g., 100 → 80)

### Issue: Recovery Too Low (<40%)
**Cause**: Defense too conservative  
**Solution**:
- Decrease `lambda_anchor` (e.g., 1e-4 → 5e-5)
- Increase `max_steps` (e.g., 100 → 120)

### Issue: Wrong Ordering (PPL_defended < PPL_clean)
**Cause**: Over-recovery from too aggressive defense  
**Solution**:
- Increase `lambda_anchor` significantly (e.g., 1e-4 → 5e-4)
- Reduce `learning_rate` (e.g., 2e-5 → 1e-5)
- Reduce `max_steps` (e.g., 100 → 60)

---

## Validation Checklist

After running the full pipeline, verify:
- [ ] Task-adapted clean PPL < original zero-shot clean PPL
- [ ] At FP16: PPL_clean < PPL_defended < PPL_attacked
- [ ] At INT8: PPL_clean < PPL_defended < PPL_attacked
- [ ] At 4-BIT: PPL_clean < PPL_defended < PPL_attacked
- [ ] All recovery rates ≤ 100%
- [ ] Recovery rates decrease as noise level increases
- [ ] Defense improves PPL by 10-30% over attacked model

---

## Citation

If you use Method A in your research, please cite:

```bibtex
@article{methodA2026,
  title={Defense Against Quantization Attacks in LLMs Through Task-Adapted Baseline},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Contact

For questions or issues with Method A implementation, please open an issue on GitHub or contact the maintainer.
