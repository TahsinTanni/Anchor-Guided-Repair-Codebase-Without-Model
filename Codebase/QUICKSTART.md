# Quick Start Guide: Realistic Defense

## TL;DR - Run Defense in 3 Commands

```bash
# 1. Run defense on attacked model (10% noise)
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct \
    --lambda_anchor 2e-4 \
    --max_steps 60

# 2. Evaluate defended model
python evaluate_defended_model.py \
    --defended_path saved_models/defended_realistic_10pct

# 3. Verify results (check that PPL_clean < PPL_defended < PPL_attacked)
```

---

## What Changed?

### Before (Unrealistic)
```bash
# Required clean model (user wouldn't have this)
python defense_anchor_finetune.py \
    --clean_model_path saved_models/wikitext2_clean_fp16 \  ❌
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    ...
```

### Now (Realistic)
```bash
# Only needs attacked model (realistic scenario)
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \  ✓
    ...
```

**Key difference:** Defense anchors to ATTACKED weights (not clean).

---

## Expected Results

```
PPL_clean    = 12.50  (original model, sanity reference)
PPL_attacked = 25.00  (downloaded vulnerable model)
PPL_defended = 17.50  (after defense, 60% recovery)

✓ Target inequality: PPL_clean < PPL_defended < PPL_attacked
✓ Partial recovery: 60-80% improvement
✓ No over-recovery: Defended worse than clean
```

---

## Batch Processing (All Attack Levels)

```bash
# Run defense for 5%, 10%, 15%, 20% attacks
python run_realistic_defense.py
```

---

## Hyperparameter Tuning

### Too much recovery (defended ≈ clean)?
```bash
# Increase anchoring, decrease steps
--lambda_anchor 3e-4 --max_steps 40
```

### Too little recovery (defended ≈ attacked)?
```bash
# Decrease anchoring, increase steps
--lambda_anchor 1e-4 --max_steps 100
```

---

## Full Documentation

- **Methodology:** [REALISTIC_DEFENSE_README.md](REALISTIC_DEFENSE_README.md)
- **Implementation:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Comparison:** [METHODOLOGY_COMPARISON.md](METHODOLOGY_COMPARISON.md)
- **Commands:** [DEFENSE_COMMANDS.sh](DEFENSE_COMMANDS.sh)

---

## Key Principle

> **The attacked model is both the starting point AND the anchor reference.**
> 
> This models a realistic scenario where a user only has access to the
> attacked/vulnerable model and wants to repair it without clean weights.
