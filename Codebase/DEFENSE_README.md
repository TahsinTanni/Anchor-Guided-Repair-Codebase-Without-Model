# Defense Against Quantization Attacks Using Weight Anchoring

This directory contains the implementation of a defense mechanism against weight-noise attacks on quantized Large Language Models (LLMs).

## Overview

### The Attack
Weight-noise attacks inject Gaussian noise into model weights to degrade performance:
- Noise is calibrated to achieve target perplexity increases (5%, 10%, 15%, 20%)
- Attack is particularly effective against quantized models (INT8, 4-BIT)
- Clean model is corrupted: `W_attacked = W_clean + N(0, σ²)`

### The Defense
**Noise-Aware Fine-Tuning with Weight Anchoring** recovers model performance by:
1. Fine-tuning the attacked model on WikiText-2 training data
2. Adding an L2 anchoring penalty to keep weights close to the clean model
3. Balancing noise removal with preservation of original model knowledge

### Loss Function
```
L_total = L_CE + λ * Σ mean((W - W_clean)²)
```

Where:
- `L_CE`: Standard cross-entropy loss for causal language modeling
- `λ`: Anchoring weight (hyperparameter, typically 1e-3)
- `W`: Current model weights (trainable)
- `W_clean`: Original clean model weights (frozen reference)
- `Σ`: Sum over selected major weight matrices (attention & MLP layers)

## Why This Defense Works

1. **Noise Correction**: Fine-tuning on clean data helps remove the injected noise
2. **Prevents Overfitting**: Anchoring penalty prevents the model from drifting too far
3. **Maintains Clean Model Quality**: L2 regularization to clean weights preserves original knowledge
4. **Quantization Robustness**: Defended weights are more stable when re-quantized

## Usage

### Step 1: Run the Defense

Defend an attacked model using weight anchoring:

```bash
python defense_anchor_finetune.py \
    --clean_model_path saved_models/wikitext2_clean_fp16 \
    --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
    --output_dir saved_models/defended_noisy_fp16_10pct \
    --lambda_anchor 1e-3 \
    --max_steps 1000 \
    --lr 2e-5 \
    --batch_size 1 \
    --grad_accum_steps 4
```

**Key Arguments:**
- `--clean_model_path`: Path to original clean FP16 model (anchor reference)
- `--attacked_model_path`: Path to noisy/attacked model to defend
- `--output_dir`: Where to save the defended model
- `--lambda_anchor`: Weight of anchoring penalty (try: 1e-4, 1e-3, 1e-2)
- `--max_steps`: Training steps (1000-2000 recommended for short defense)
- `--lr`: Learning rate (2e-5 is conservative)

### Step 2: Evaluate Defended Model

Use the evaluation helper script:

```bash
python evaluate_defended_model.py \
    --model_path saved_models/defended_noisy_fp16_10pct \
    --output_file defended_10pct_results.json
```

This will test the defended model at:
- FP16 precision
- INT8 quantization
- 4-BIT quantization

### Step 3: Compare Results

Compare perplexity across:
- **Clean model** (baseline)
- **Attacked model** (with noise)
- **Defended model** (after anchoring fine-tuning)

Expected outcome: Defended model perplexity should be closer to clean model than attacked model.

## Hyperparameter Tuning

### Lambda (Anchoring Weight)

The `lambda_anchor` parameter controls the trade-off between adaptation and anchoring:

| Lambda | Effect | When to Use |
|--------|--------|-------------|
| 1e-4 | More aggressive fine-tuning | High noise levels (15-20%) |
| 1e-3 | Balanced (default) | Medium noise levels (10%) |
| 1e-2 | Strong anchoring | Low noise levels (5%), preserve clean model |

### Training Steps

- **Too few** (< 500): May not fully remove noise
- **Optimal** (1000-2000): Good balance
- **Too many** (> 3000): Risk of overfitting to WikiText-2

### Learning Rate

- **2e-5** (default): Conservative, stable
- **5e-5**: Faster convergence, higher risk
- **1e-5**: Very conservative, slower

## Example Workflow

```bash
# 1. Create attacked model (already done in your analysis)
# You have: saved_models/wikitext2_noisy_fp16_10pct/

# 2. Defend with different lambda values
for lambda in 1e-4 1e-3 1e-2; do
    python defense_anchor_finetune.py \
        --clean_model_path saved_models/wikitext2_clean_fp16 \
        --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \
        --output_dir saved_models/defended_10pct_lambda${lambda} \
        --lambda_anchor ${lambda} \
        --max_steps 1000
done

# 3. Evaluate all defended models
for lambda in 1e-4 1e-3 1e-2; do
    python evaluate_defended_model.py \
        --model_path saved_models/defended_10pct_lambda${lambda} \
        --output_file defended_10pct_lambda${lambda}_results.json
done

# 4. Compare results
# Check which lambda gives best recovery
```

## File Structure

```
defense_anchor_finetune.py      # Main defense script
evaluate_defended_model.py       # Evaluation helper
DEFENSE_README.md               # This file
saved_models/
  ├── wikitext2_clean_fp16/              # Clean model (anchor)
  ├── wikitext2_noisy_fp16_10pct/        # Attacked model
  └── defended_noisy_fp16_10pct/         # Defended model output
      ├── config.json
      ├── model.safetensors
      ├── tokenizer files...
      └── defense_stats.json              # Training statistics
```

## Implementation Details

### Anchored Parameters

The defense anchors major weight matrices while skipping:
- **Anchored**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2`
- **Not Anchored**: Biases, LayerNorm weights, embeddings, LM head

This focuses the regularization on the most critical parameters.

### Training Configuration

- **Optimizer**: AdamW with weight decay 0.01
- **Gradient Clipping**: 1.0 (prevents instability)
- **FP16 Training**: Enabled by default for efficiency
- **Batch Size**: 1 with 4 gradient accumulation steps (effective batch size = 4)
- **Warmup**: 50 steps
- **Sequence Length**: 2048 (matches perplexity evaluation)

### Dataset Preprocessing

Uses the same tokenization as perplexity scripts:
- WikiText-2 train split
- Grouped into blocks of 2048 tokens
- Causal LM objective (predict next token)

## Expected Results

For a 10% noise attack (σ=0.000838):

| Model Type | FP16 PPL | INT8 PPL | 4-BIT PPL |
|------------|----------|----------|-----------|
| Clean | 12.46 | 12.48 | 13.79 |
| Attacked | 13.20 | 13.43 | 14.63 |
| Defended | ~12.6-12.8 | ~12.7-12.9 | ~14.0-14.2 |

Defense should recover 50-80% of the perplexity degradation.

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 1
- Increase `--grad_accum_steps` to 8
- Reduce `--seq_len` to 1024

### Training Unstable
- Reduce `--lr` to 1e-5
- Increase `--lambda_anchor` to 1e-2
- Enable `--fp16` if not already

### Poor Defense Results
- Try different `--lambda_anchor` values
- Increase `--max_steps` to 2000
- Check if attacked model is too corrupted (noise > 20%)

## Citation

If you use this defense mechanism, please cite:
```bibtex
@misc{defense_weight_anchoring_2026,
  title={Defense Against Quantization Attacks Using Weight Anchoring},
  author={Your Name},
  year={2026},
  note={Noise-aware fine-tuning with L2 regularization to clean weights}
}
```

## References

- Weight noise injection attacks on quantized LLMs
- L2 regularization and knowledge distillation
- Robust fine-tuning techniques for corrupted models
