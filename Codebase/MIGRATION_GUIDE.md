# Migration Guide: From Clean-Anchored to Attacked-Anchored Defense

## Overview

This guide helps you migrate from the old clean-model-anchored defense to the new realistic attacked-model-anchored defense.

---

## What's Different?

### Core Concept Change

| Aspect | Old | New |
|--------|-----|-----|
| Anchor reference | Clean model weights | Attacked model weights |
| User access | Clean + Attacked models | Attacked model only |
| Required files | 2 models | 1 model |
| Recovery type | Full (>100%) | Partial (60-80%) |
| Scenario | Unrealistic | Realistic |

---

## Command Changes

### Old Command Structure
```bash
python defense_anchor_finetune.py \
    --clean_model_path saved_models/wikitext2_clean_fp16 \      # REMOVED
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_noisy_fp16_10pct \
    --lambda_anchor 1e-3 \                                       # CHANGED
    --max_steps 1000                                             # CHANGED
```

### New Command Structure
```bash
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct \
    --lambda_anchor 2e-4 \                                       # NEW DEFAULT
    --max_steps 60                                               # NEW DEFAULT
```

---

## Parameter Migration Table

| Parameter | Old Default | New Default | Change Reason |
|-----------|-------------|-------------|---------------|
| `--clean_model_path` | **Required** | **Removed** | Not needed in realistic scenario |
| `--lambda_anchor` | `1e-3` | `2e-4` | Weaker anchoring allows repair |
| `--max_steps` | `1000` | `60` | Shorter training prevents over-recovery |
| `--eval_steps` | `100` | `20` | More frequent monitoring |
| `--save_steps` | `500` | `30` | Earlier checkpointing |
| `--warmup_steps` | `50` | `10` | Less warmup needed |
| `--grad_accum_steps` | `4` | `2` | Smaller effective batch size |

---

## Re-Running Defenses

### Option 1: Re-run All Defenses (Recommended)

If you want to follow the new realistic methodology:

```bash
# Clean up old defended models (optional, for clarity)
rm -rf saved_models/defended_noisy_fp16_*

# Run new realistic defense for all attack levels
python run_realistic_defense.py
```

This will create:
- `saved_models/defended_realistic_5pct/`
- `saved_models/defended_realistic_10pct/`
- `saved_models/defended_realistic_15pct/`
- `saved_models/defended_realistic_20pct/`

### Option 2: Keep Old Results, Run New Alongside

If you want to compare old vs. new:

```bash
# Run new defense with different output directory
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct_NEW \
    --lambda_anchor 2e-4 \
    --max_steps 60
```

This preserves old results in `saved_models/defended_noisy_fp16_10pct/` while creating new results in `saved_models/defended_realistic_10pct_NEW/`.

---

## Updating Evaluation Scripts

### If You Modified `evaluate_defended_model.py`

The evaluation script should work without changes, but if you hardcoded paths:

**Old:**
```python
clean_path = "saved_models/wikitext2_clean_fp16"
attacked_path = "saved_models/wikitext2_noisy_fp16_10pct"
defended_path = "saved_models/defended_noisy_fp16_10pct"  # Old
```

**New:**
```python
clean_path = "saved_models/wikitext2_clean_fp16"
attacked_path = "saved_models/wikitext2_noisy_fp16_10pct"
defended_path = "saved_models/defended_realistic_10pct"  # New
```

### Clean Model Still Used for Evaluation

**Important:** The clean model is still needed for evaluation as a sanity reference!

```bash
python evaluate_defended_model.py \
    --clean_path saved_models/wikitext2_clean_fp16 \      # Still needed
    --attacked_path saved_models/wikitext2_noisy_fp16_10pct \
    --defended_path saved_models/defended_realistic_10pct
```

---

## Updating Plotting Scripts

If you created plots with old results, update the paths:

**Old:**
```python
defended_dirs = [
    "saved_models/defended_noisy_fp16_5pct",
    "saved_models/defended_noisy_fp16_10pct",
    "saved_models/defended_noisy_fp16_15pct",
    "saved_models/defended_noisy_fp16_20pct",
]
```

**New:**
```python
defended_dirs = [
    "saved_models/defended_realistic_5pct",
    "saved_models/defended_realistic_10pct",
    "saved_models/defended_realistic_15pct",
    "saved_models/defended_realistic_20pct",
]
```

---

## Expected Results Changes

### Old Results (Clean-Anchored)
```
Attack Level: 10%
PPL_clean    : 12.50
PPL_attacked : 25.00
PPL_defended : 11.80  ❌ Over-recovery (beats clean!)

Recovery: >100% ❌
```

### New Results (Attacked-Anchored)
```
Attack Level: 10%
PPL_clean    : 12.50
PPL_attacked : 25.00
PPL_defended : 17.50  ✓ Partial recovery

Recovery: 60% ✓
Target inequality: PPL_clean < PPL_defended < PPL_attacked ✓
```

---

## Troubleshooting Migration Issues

### Issue 1: Script Errors About Missing `--clean_model_path`

**Old Error:**
```
error: the following arguments are required: --clean_model_path
```

**Solution:**
Update your command to remove `--clean_model_path`:
```bash
# Remove this line:
--clean_model_path saved_models/wikitext2_clean_fp16 \
```

### Issue 2: Different Results Than Before

**This is expected!** The new methodology produces different results:
- Old: Over-recovery (PPL_defended ≈ PPL_clean)
- New: Partial recovery (PPL_clean < PPL_defended < PPL_attacked)

### Issue 3: Existing Scripts Reference Old Directories

**Solution:**
Either:
1. Update directory names in scripts
2. Create symbolic links (Unix/Linux):
```bash
ln -s saved_models/defended_realistic_10pct saved_models/defended_noisy_fp16_10pct
```
3. Or on Windows (PowerShell as admin):
```powershell
New-Item -ItemType SymbolicLink `
    -Path "saved_models\defended_noisy_fp16_10pct" `
    -Target "saved_models\defended_realistic_10pct"
```

---

## Verification Checklist

After migration, verify:

- [ ] Defense script runs without `--clean_model_path`
- [ ] Defended models created in new directories
- [ ] Evaluation scripts updated with new paths
- [ ] Plotting scripts updated with new paths
- [ ] Results show partial recovery (60-80%)
- [ ] Target inequality satisfied: PPL_clean < PPL_defended < PPL_attacked
- [ ] Documentation updated with new methodology

---

## Side-by-Side Comparison Script

Create a script to compare old vs. new results:

```python
"""Compare old clean-anchored vs. new attacked-anchored results."""

import json

def load_ppl(path):
    """Load perplexity from results JSON."""
    with open(f"{path}/perplexity_results.json") as f:
        return json.load(f)["perplexity"]

# Load results
clean_ppl = load_ppl("saved_models/wikitext2_clean_fp16")
attacked_ppl = load_ppl("saved_models/wikitext2_noisy_fp16_10pct")
old_defended_ppl = load_ppl("saved_models/defended_noisy_fp16_10pct")
new_defended_ppl = load_ppl("saved_models/defended_realistic_10pct")

print("PPL Comparison (10% Attack)")
print("=" * 60)
print(f"Clean         : {clean_ppl:.2f}")
print(f"Attacked      : {attacked_ppl:.2f}")
print(f"Defended (old): {old_defended_ppl:.2f}")
print(f"Defended (new): {new_defended_ppl:.2f}")
print("=" * 60)

# Calculate recoveries
old_recovery = (attacked_ppl - old_defended_ppl) / (attacked_ppl - clean_ppl) * 100
new_recovery = (attacked_ppl - new_defended_ppl) / (attacked_ppl - clean_ppl) * 100

print(f"\nOld recovery: {old_recovery:.1f}% {'❌ Over-recovery' if old_recovery > 100 else ''}")
print(f"New recovery: {new_recovery:.1f}% {'✓ Partial recovery' if 60 <= new_recovery <= 80 else ''}")

# Check target inequality
old_valid = clean_ppl < old_defended_ppl < attacked_ppl
new_valid = clean_ppl < new_defended_ppl < attacked_ppl

print(f"\nTarget inequality (Clean < Defended < Attacked):")
print(f"Old: {'✓' if old_valid else '❌'}")
print(f"New: {'✓' if new_valid else '❌'}")
```

---

## Complete Migration Example

### Step 1: Backup Old Results (Optional)
```bash
mkdir -p backup_old_defense
cp -r saved_models/defended_noisy_fp16_* backup_old_defense/
```

### Step 2: Run New Defense
```bash
python run_realistic_defense.py
```

### Step 3: Update Evaluation
```bash
# Update paths in your evaluation scripts
sed -i 's/defended_noisy_fp16/defended_realistic/g' evaluate_*.py

# Or manually update:
# defended_noisy_fp16_10pct → defended_realistic_10pct
```

### Step 4: Verify Results
```bash
python evaluate_defended_model.py \
    --defended_path saved_models/defended_realistic_10pct
```

### Step 5: Update Plots
```bash
# Update paths in plotting scripts
sed -i 's/defended_noisy_fp16/defended_realistic/g' plot_*.py

# Regenerate plots
python plot_defense_comparison.py
```

---

## Summary

**Migration is simple:**
1. Remove `--clean_model_path` from commands
2. Use new hyperparameter defaults
3. Update output directory names
4. Verify partial recovery results

**Key takeaway:** The new methodology is more realistic and requires fewer inputs (no clean model needed for defense).

---

## Need Help?

If you encounter issues during migration:

1. Check [REALISTIC_DEFENSE_README.md](REALISTIC_DEFENSE_README.md)
2. Review [METHODOLOGY_COMPARISON.md](METHODOLOGY_COMPARISON.md)
3. See example commands in [DEFENSE_COMMANDS.sh](DEFENSE_COMMANDS.sh)
4. Verify installation with [QUICKSTART.md](QUICKSTART.md)
