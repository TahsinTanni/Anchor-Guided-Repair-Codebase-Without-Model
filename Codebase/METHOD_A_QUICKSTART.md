# Method A Quick Start Guide

## TL;DR

Run the complete pipeline:
```bash
# Linux/Mac
bash run_method_a_pipeline.sh

# Windows PowerShell
.\run_method_a_pipeline.ps1
```

Or run individual steps:

```bash
# Step 1: Train task-adapted clean baseline (one-time, ~30-60 min)
python train_task_adapted_clean.py \
    --output_dir saved_models/wikitext2_clean_adapted_fp16 \
    --max_steps 200

# Step 2: Generate attacked models (~10 min)
python weight_noise_from_adapted.py \
    --base_model_path saved_models/wikitext2_clean_adapted_fp16 \
    --noise_levels 5 10 15 20

# Step 3: Defend attacked models (~20-30 min per noise level)
python defense_anchor_finetune_adapted.py \
    --clean_model_path saved_models/wikitext2_clean_adapted_fp16 \
    --attacked_model_path saved_models/wikitext2_attacked_adapted_fp16_10pct \
    --output_dir saved_models/wikitext2_defended_adapted_fp16_10pct \
    --lambda_anchor 1e-4 \
    --max_steps 100

# Step 4: Evaluate defended model (~5-10 min)
python evaluate_defended_adapted.py \
    --clean_model_path saved_models/wikitext2_clean_adapted_fp16 \
    --attacked_model_path saved_models/wikitext2_attacked_adapted_fp16_10pct \
    --defended_model_path saved_models/wikitext2_defended_adapted_fp16_10pct \
    --output_file defended_adapted_10pct_results.json
```

## What You Get

### Input
- Original Qwen-0.5B model (zero-shot on WikiText-2)

### Output
- **Task-adapted clean baseline** with lower PPL
- **Attacked models** at 5%, 10%, 15%, 20% noise
- **Defended models** with bounded recovery (≤100%)
- **Evaluation results** with proper baseline comparison

### Expected Results
```
PPL_clean_adapted < PPL_defended < PPL_attacked
Recovery: 60-80% typical (bounded at 100%)
```

## File Overview

### New Scripts (Method A)
1. **train_task_adapted_clean.py** - Creates task-adapted baseline
2. **weight_noise_from_adapted.py** - Attacks adapted model
3. **defense_anchor_finetune_adapted.py** - Defends with clean anchoring
4. **evaluate_defended_adapted.py** - Evaluates with proper baseline
5. **run_method_a_pipeline.sh/.ps1** - Automated pipeline

### Documentation
- **METHOD_A_README.md** - Complete methodology documentation
- **METHOD_A_QUICKSTART.md** - This file

## Time Estimates (RTX 3060 12GB)

| Step | Duration | Can Skip? |
|------|----------|-----------|
| Step 1: Task adaptation | 30-60 min | After first run |
| Step 2: Attack generation | 10 min | No (quick) |
| Step 3: Defense (per noise level) | 20-30 min | No |
| Step 4: Evaluation (per noise level) | 5-10 min | No |
| **Total (all 4 noise levels)** | **2-3 hours** | - |

## Hyperparameter Quick Reference

### Task Adaptation (Step 1)
```bash
--max_steps 200              # Sufficient for adaptation
--learning_rate 2e-5         # Conservative
--gradient_accumulation_steps 4
--fp16                       # Memory efficiency
```

### Defense (Step 3)
```bash
--lambda_anchor 1e-4         # Good balance (recommended)
--max_steps 100              # Moderate training
--learning_rate 2e-5         # Conservative
--gradient_accumulation_steps 2
```

### Lambda Tuning
- **Too low (<1e-5)**: Over-recovery (PPL_defended < PPL_clean)
- **Recommended (1e-4)**: Balanced (~70% recovery)
- **Too high (>5e-4)**: Under-recovery (<50% recovery)

## Troubleshooting

### OOM Errors
```bash
# Reduce sequence length
--seq_len 1024

# Reduce gradient accumulation
--gradient_accumulation_steps 2
```

### Recovery > 100%
```bash
# Increase lambda (more conservative)
--lambda_anchor 2e-4

# Reduce training steps
--max_steps 80
```

### Recovery Too Low (<40%)
```bash
# Decrease lambda (more aggressive)
--lambda_anchor 5e-5

# Increase training steps
--max_steps 120
```

## Validation Checklist

After running the pipeline:
- [ ] Task-adapted PPL < zero-shot PPL
- [ ] Ordering correct at FP16, INT8, 4-BIT
- [ ] All recovery rates ≤ 100%
- [ ] Recovery decreases with noise level
- [ ] Defense improves PPL by 10-30%

## Next Steps

1. **Run the pipeline**: Use automated script or manual steps
2. **Check results**: Verify JSON output files
3. **Adjust if needed**: Tune lambda_anchor based on results
4. **Generate plots**: Create visualization of results
5. **Compare**: Method A vs original methodology

## Full Documentation

See [METHOD_A_README.md](METHOD_A_README.md) for complete documentation including:
- Detailed methodology explanation
- Complete hyperparameter guide
- Comparison with original approach
- Directory structure
- Advanced troubleshooting

## Support

For issues or questions:
1. Check [METHOD_A_README.md](METHOD_A_README.md) troubleshooting section
2. Verify GPU memory (requires ~10GB for RTX 3060)
3. Check logs in `saved_models/*/logs/`
4. Open GitHub issue with error details
