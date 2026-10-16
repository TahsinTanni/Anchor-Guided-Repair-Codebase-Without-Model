# Anchor-Guided Repair of Weight-Noise-Attacked LLMs

Code, results, and figures for a thesis project on **repairing language models whose weights have been corrupted by Gaussian noise injection**, and on how that damage and the repair carry over to quantized (INT8 / 4-bit) versions of the model.

The main model is **Qwen2.5-0.5B-Instruct**, evaluated by perplexity (PPL) on the **WikiText-2** test set (4,358 samples). The model weights themselves are **not included** in this repository (hence "Without-Model"). The scripts regenerate them under `Codebase/saved_models/`.

---

## Idea

1. **Attack.** Add Gaussian noise `N(0, σ²)` to the weights of a clean model. σ is calibrated so the FP16 perplexity rises by a target amount (5%, 10%, 15%, 20%).
2. **Quantize.** Evaluate the clean and attacked models at FP16, INT8, and 4-bit NF4 (bitsandbytes). This shows how the damage survives quantization.
3. **Repair (anchor-guided fine-tuning).** Fine-tune the attacked model on clean WikiText-2 text with an L2 penalty that keeps the weights close to an anchor:

   ```
   Loss = LM_loss(WikiText-2) + λ_anchor · ‖W − W_anchor‖²
   ```

   - **Realistic setting (main method):** the anchor is the *attacked* model itself. The user only has the downloaded, corrupted model, and the clean weights are used only to evaluate the result.
   - **Method A (comparison):** a task-adapted clean model serves as the baseline and anchor. This is an idealized setting where clean weights are available.

---

## Repository layout

```
Codebase/
├── Baseline & calibration
│   ├── calculate_perplexity*.py            # PPL of the clean model (FP16 / INT8 / 4-bit)
│   ├── weight_noise_injection.py           # sweeps σ, fits r ≈ exp(a·σ²), solves for σ*
│   └── calculate_optimal_sigmas_wikitext2.py
├── Attack
│   ├── wikitext2_full_analysis.py          # builds noisy models at each level + quantized eval
│   ├── quantize_noisy_model.py
│   └── weight_noise_from_adapted.py        # Method A: attack the task-adapted model
├── Defense
│   ├── defense_anchor_finetune.py          # anchor-guided repair (realistic, attacked anchor)
│   ├── run_realistic_defense.py            # batch over 5/10/15/20% noise
│   ├── defense_anchor_finetune_adapted.py  # Method A defense
│   ├── train_task_adapted_clean.py         # Method A clean baseline
│   └── run_method_a_pipeline.sh
├── Evaluation
│   ├── evaluate_defended_model.py
│   ├── evaluate_defended_adapted.py
│   └── defend_and_evaluate_all.py
├── Plotting: plot_*.py
├── Results: *.json, *.txt, *.png
└── Docs: QUICKSTART.md, REALISTIC_DEFENSE_README.md, METHOD_A_README.md, DOCS_INDEX.md, ...

thesis pic/
├── attack part/                     # PPL vs. noise level plots
├── def mecha for both models/       # defense results for Qwen (v1–v3) and Pythia (v1–v2)
└── sanity checking/                 # ordering, category, and stability-variance checks
```

`Codebase/DOCS_INDEX.md` lists the more detailed method documents.

---

## Setup

Requirements: Python 3.10+ and a CUDA GPU. The experiments were run on a GPU with about 12 GB of memory, and bitsandbytes quantization needs CUDA.

```bash
pip install torch transformers datasets bitsandbytes accelerate numpy matplotlib tqdm
```

The model (`Qwen/Qwen2.5-0.5B-Instruct`) and the dataset (`wikitext-2-raw-v1`) download automatically from Hugging Face on first run.

---

## Running the pipeline

Run these commands from inside `Codebase/`:

```bash
# 1. Clean baseline perplexity
python calculate_perplexity_wikitext2.py

# 2. Calibrate noise levels (writes weight_noise_injection_wikitext2_results.json)
python weight_noise_injection.py

# 3. Build attacked models and evaluate them at FP16 / INT8 / 4-bit
python wikitext2_full_analysis.py

# 4. Repair an attacked model (anchored to the attacked weights)
python defense_anchor_finetune.py \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_realistic_10pct \
    --lambda_anchor 2e-4 \
    --max_steps 60

#    ...or every noise level at once
python run_realistic_defense.py

# 5. Evaluate the repaired model
python evaluate_defended_model.py \
    --model_path saved_models/defended_realistic_10pct \
    --output_file defended_realistic_10pct_results.json
```

Main defense hyperparameters (defaults in `defense_anchor_finetune.py`): `--lambda_anchor 2e-4`, `--lr 2e-5`, `--max_steps 60`, `--seq_len 2048`, `--batch_size 1`, `--grad_accum_steps 2`, `--fp16`.

- To recover **more**, lower `λ_anchor` or raise `max_steps`.
- To recover **less**, do the opposite.

Method A runs end to end with `bash run_method_a_pipeline.sh` (see `METHOD_A_QUICKSTART.md`).

---

## Results (Qwen2.5-0.5B-Instruct, WikiText-2 test PPL)

**Calibrated noise levels.** Fitted model: `r ≈ exp(135679 · σ²)`

| Target PPL increase | σ*       |
|---------------------|----------|
| 5%                  | 0.000600 |
| 10%                 | 0.000838 |
| 15%                 | 0.001015 |
| 20%                 | 0.001159 |

**Clean vs. attacked models** (`noisy_quantized_perplexity_wikitext2.txt`)

| Model        | FP16  | INT8  | 4-bit |
|--------------|-------|-------|-------|
| Clean        | 12.46 | 12.48 | 13.79 |
| Attacked 5%  | 12.78 | 12.81 | 14.14 |
| Attacked 10% | 13.20 | 13.43 | 14.63 |
| Attacked 15% | 13.79 | 13.53 | 15.31 |
| Attacked 20% | 13.80 | 13.69 | 15.17 |

**After anchor-guided repair, 10% attack, realistic setting** (`defended_realistic_10pct_v*_results.json`)

| Run       | FP16  | INT8  | 4-bit |
|-----------|-------|-------|-------|
| v1 / v3   | 12.02 | 12.08 | 13.22 |
| v2 / v4–5 | 12.08 | 12.14 | 13.31 |
| v6        | 12.21 | 12.28 | 13.46 |
| v7        | 12.21 | 12.28 | 13.51 |
| v8        | 12.13 | 12.19 | 13.56 |

The repair brings perplexity back below the attacked level at every precision. In these runs the repaired model also ends up slightly *below* the clean baseline. Short fine-tuning on WikiText-2 adapts the model to the evaluation domain, so read these numbers relative to a task-adapted baseline (this is the motivation for Method A). Earlier runs anchored to the clean weights are in `defended_10pct_v*_results.json` and `defended_20pct_v2_results.json`.

Figures for every experiment are in `Codebase/*.png` and `thesis pic/`.
