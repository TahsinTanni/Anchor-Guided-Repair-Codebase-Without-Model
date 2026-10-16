# Method A: Task-Adapted Clean Baseline - Documentation Index

## Quick Navigation

### 🚀 Getting Started
1. **[Quick Start Guide](METHOD_A_QUICKSTART.md)** - Start here for TL;DR and quick commands
2. **[Implementation Summary](METHOD_A_IMPLEMENTATION_SUMMARY.md)** - High-level overview of what was built
3. **[Complete README](METHOD_A_README.md)** - Full methodology and detailed documentation

### 📜 Core Scripts

| Script | Purpose | Duration | Documentation Link |
|--------|---------|----------|-------------------|
| `train_task_adapted_clean.py` | Create task-adapted baseline | 30-60 min | [README §Step 1](METHOD_A_README.md#step-1-train-task-adapted-clean-baseline-one-time) |
| `weight_noise_from_adapted.py` | Generate attacked models | 10 min | [README §Step 2](METHOD_A_README.md#step-2-generate-attacked-models-from-task-adapted-baseline) |
| `defense_anchor_finetune_adapted.py` | Defend with clean anchoring | 20-30 min | [README §Step 3](METHOD_A_README.md#step-3-defend-attacked-models-anchor-to-task-adapted-clean) |
| `evaluate_defended_adapted.py` | Evaluate with proper baseline | 5-10 min | [README §Step 4](METHOD_A_README.md#step-4-evaluate-with-proper-baseline-comparison) |

### 🤖 Automation Scripts

| Script | Platform | Usage |
|--------|----------|-------|
| `run_method_a_pipeline.sh` | Linux/Mac | `bash run_method_a_pipeline.sh` |
| `run_method_a_pipeline.ps1` | Windows | `.\run_method_a_pipeline.ps1` |

### 📖 Documentation Files

| File | Content | When to Read |
|------|---------|--------------|
| [METHOD_A_QUICKSTART.md](METHOD_A_QUICKSTART.md) | Quick reference, TL;DR commands | **Start here** - First time users |
| [METHOD_A_IMPLEMENTATION_SUMMARY.md](METHOD_A_IMPLEMENTATION_SUMMARY.md) | What was built, how it works | Understanding the implementation |
| [METHOD_A_README.md](METHOD_A_README.md) | Complete methodology guide | Deep dive, troubleshooting |
| [METHOD_A_INDEX.md](METHOD_A_INDEX.md) | This file - navigation hub | Finding specific information |

## By Task

### I want to...

#### Run the complete pipeline
→ **[Quick Start](METHOD_A_QUICKSTART.md#tldr)** or **[Automation Scripts](METHOD_A_README.md#complete-pipeline-example)**

#### Understand the methodology
→ **[Implementation Summary](METHOD_A_IMPLEMENTATION_SUMMARY.md#problem-solved)** or **[README Overview](METHOD_A_README.md#overview)**

#### Tune hyperparameters
→ **[Hyperparameters Summary](METHOD_A_README.md#hyperparameters-summary)** or **[Lambda Guidelines](METHOD_A_README.md#lambda-anchor-guidelines)**

#### Fix OOM errors
→ **[OOM Safety](METHOD_A_README.md#oom-safety-rtx-3060-12gb)** or **[Troubleshooting](METHOD_A_QUICKSTART.md#oom-errors)**

#### Fix recovery >100%
→ **[Troubleshooting §Recovery](METHOD_A_README.md#issue-recovery--100)** or **[Quick Fix](METHOD_A_QUICKSTART.md#recovery--100)**

#### Understand the difference from the original
→ **[Comparison Table](METHOD_A_README.md#comparison-old-vs-new-methodology)** or **[Key Differences](METHOD_A_IMPLEMENTATION_SUMMARY.md#key-differences-from-original-approach)**

#### See expected results
→ **[Expected Results](METHOD_A_README.md#expected-results)** or **[Expected Behavior](METHOD_A_IMPLEMENTATION_SUMMARY.md#expected-behavior)**

#### Validate my results
→ **[Validation Checklist](METHOD_A_README.md#validation-checklist)** or **[Quick Checklist](METHOD_A_QUICKSTART.md#validation-checklist)**

## By Use Case

### First-Time User
1. Read: [METHOD_A_QUICKSTART.md](METHOD_A_QUICKSTART.md)
2. Run: `bash run_method_a_pipeline.sh` or `.\run_method_a_pipeline.ps1`
3. Validate: [Validation Checklist](METHOD_A_QUICKSTART.md#validation-checklist)

### Researcher/Developer
1. Read: [METHOD_A_IMPLEMENTATION_SUMMARY.md](METHOD_A_IMPLEMENTATION_SUMMARY.md)
2. Deep dive: [METHOD_A_README.md](METHOD_A_README.md)
3. Customize: See [Hyperparameters Summary](METHOD_A_README.md#hyperparameters-summary)

### Troubleshooting
1. Quick fixes: [Quick Start §Troubleshooting](METHOD_A_QUICKSTART.md#troubleshooting)
2. Detailed guide: [README §Troubleshooting](METHOD_A_README.md#troubleshooting)
3. OOM issues: [README §OOM Safety](METHOD_A_README.md#oom-safety-rtx-3060-12gb)

## Workflow Diagram

```
Original Clean Model (Zero-Shot)
         ↓
    STEP 1: train_task_adapted_clean.py
         ↓
Task-Adapted Clean Baseline (Fine-Tuned)
         ↓
    STEP 2: weight_noise_from_adapted.py
         ↓
Attacked Models (5%, 10%, 15%, 20%)
         ↓
    STEP 3: defense_anchor_finetune_adapted.py
         ↓
Defended Models
         ↓
    STEP 4: evaluate_defended_adapted.py
         ↓
Results (PPL + Recovery Rates)
```

## Key Concepts

### Task-Adapted Clean Baseline
- **What**: Clean model fine-tuned on WikiText-2
- **Why**: Fair baseline for comparison
- **How**: Standard LM training (no anchoring)
- **Where**: [README §Overview](METHOD_A_README.md#overview)

### Attack Generation
- **What**: Inject Gaussian noise into task-adapted clean
- **Why**: Create realistic attacked models
- **How**: Noise proportional to target PPL increase
- **Where**: [README §Step 2](METHOD_A_README.md#step-2-generate-attacked-models-from-task-adapted-baseline)

### Defense with Clean Anchoring
- **What**: Fine-tune attacked model with L2 penalty to clean
- **Why**: Repair damage while staying bounded
- **How**: Loss = LM_Loss + λ × ||θ - θ_clean||²
- **Where**: [README §Step 3](METHOD_A_README.md#step-3-defend-attacked-models-anchor-to-task-adapted-clean)

### Recovery Calculation
- **Formula**: (PPL_attacked - PPL_defended) / (PPL_attacked - PPL_clean) × 100%
- **Range**: 0-100% (bounded)
- **Interpretation**: % of damage repaired
- **Where**: [README §Expected Results](METHOD_A_README.md#expected-results)

## File Locations

### Scripts
```
c:\Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection\
├── train_task_adapted_clean.py
├── weight_noise_from_adapted.py
├── defense_anchor_finetune_adapted.py
├── evaluate_defended_adapted.py
├── run_method_a_pipeline.sh
└── run_method_a_pipeline.ps1
```

### Documentation
```
c:\Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection\
├── METHOD_A_INDEX.md                      # This file
├── METHOD_A_QUICKSTART.md                 # Quick reference
├── METHOD_A_IMPLEMENTATION_SUMMARY.md     # Implementation overview
└── METHOD_A_README.md                     # Complete guide
```

### Outputs (After Running)
```
saved_models/
├── wikitext2_clean_adapted_fp16/          # Step 1
├── wikitext2_attacked_adapted_fp16_*/     # Step 2
└── wikitext2_defended_adapted_fp16_*/     # Step 3

Logs:
├── task_adapted_clean_training_log.json   # Step 1
├── attacked_models_from_adapted_log.json  # Step 2
└── defended_adapted_*_results.json        # Step 4
```

## Support

### Common Questions

**Q: Do I need to run Step 1 every time?**  
A: No, Step 1 is one-time. The script will skip it if the model already exists.

**Q: How much GPU memory do I need?**  
A: Minimum 10GB (tested on RTX 3060 12GB). See [OOM Safety](METHOD_A_README.md#oom-safety-rtx-3060-12gb).

**Q: What if recovery is >100%?**  
A: Increase `lambda_anchor` or reduce `max_steps`. See [Troubleshooting](METHOD_A_README.md#issue-recovery--100).

**Q: What if recovery is too low?**  
A: Decrease `lambda_anchor` or increase `max_steps`. See [Troubleshooting](METHOD_A_README.md#issue-recovery-too-low-40).

**Q: How long does the full pipeline take?**  
A: 2-3 hours total (RTX 3060 12GB). See [Time Estimates](METHOD_A_QUICKSTART.md#time-estimates-rtx-3060-12gb).

### Getting Help

1. **Check documentation**: Start with [Quick Start](METHOD_A_QUICKSTART.md)
2. **Review troubleshooting**: See [README §Troubleshooting](METHOD_A_README.md#troubleshooting)
3. **Validate setup**: Use [Validation Checklist](METHOD_A_QUICKSTART.md#validation-checklist)
4. **Open issue**: Provide error logs and configuration

## Version History

- **v1.0** (Current): Initial Method A implementation
  - Task-adapted clean baseline
  - Clean-anchored defense
  - Bounded recovery rates
  - Complete documentation

## Next Steps

1. **New users**: Start with [METHOD_A_QUICKSTART.md](METHOD_A_QUICKSTART.md)
2. **Researchers**: Read [METHOD_A_IMPLEMENTATION_SUMMARY.md](METHOD_A_IMPLEMENTATION_SUMMARY.md)
3. **Deep dive**: Study [METHOD_A_README.md](METHOD_A_README.md)
4. **Run pipeline**: Use automation scripts or manual commands
5. **Validate**: Check results against expected behavior

---

**Last Updated**: 2026-01-18  
**Status**: Complete and ready to use  
**License**: MIT (assumed)  
**Contact**: See main repository README
