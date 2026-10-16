"""
Train Task-Adapted Clean Baseline (Method A)

PURPOSE:
--------
This script creates a TASK-ADAPTED clean baseline by fine-tuning the original
clean Qwen-0.5B model on WikiText-2. This baseline serves as the reference point
for both attack generation and defense evaluation.

MOTIVATION:
-----------
The original clean model is zero-shot on WikiText-2, which can result in:
- Defended models achieving better PPL than the clean baseline
- Recovery rates > 100% (nonsensical)
- Unclear defense effectiveness

By creating a task-adapted clean baseline:
- PPL_clean_adapted < PPL_defended < PPL_attacked (expected ordering)
- Recovery stays bounded ≤ 100%
- Defense effectiveness is interpretable

METHODOLOGY (METHOD A):
-----------------------
1. Load original clean Qwen-0.5B checkpoint
2. Fine-tune on WikiText-2 (standard LM training, NO anchoring)
3. Save as: saved_models/wikitext2_clean_adapted_fp16
4. Use this as the baseline for:
   - Attack generation (inject noise into THIS model)
   - Defense anchoring (anchor toward THIS model)
   - Evaluation (compare against THIS model)

HYPERPARAMETERS (RTX 3060 12GB safe):
--------------------------------------
- max_steps = 200 (one-time training)
- learning_rate = 2e-5
- per_device_train_batch_size = 1 (OOM safety)
- gradient_accumulation_steps = 4
- fp16 = True
- lambda_anchor = 0 (NO anchoring - standard LM training)

USAGE:
------
python train_task_adapted_clean.py \\
    --model_name Qwen/Qwen2.5-0.5B-Instruct \\
    --output_dir saved_models/wikitext2_clean_adapted_fp16 \\
    --max_steps 200 \\
    --learning_rate 2e-5 \\
    --gradient_accumulation_steps 4

EXPECTED OUTPUT:
----------------
- saved_models/wikitext2_clean_adapted_fp16/ (model checkpoint)
- task_adapted_clean_training_log.json (training metrics)
- Final perplexity on WikiText-2 test set (should be < original clean PPL)

NEXT STEPS:
-----------
1. Generate attacked models FROM this checkpoint using weight_noise_from_adapted.py
2. Defend attacked models using defense_anchor_finetune_adapted.py
3. Evaluate using the adapted clean as the baseline reference
"""

import argparse
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from datetime import datetime
from tqdm import tqdm


def calculate_perplexity(model, tokenizer, text, desc=""):
    """Calculate perplexity on text for evaluation."""
    device = model.device
    
    print(f"\nCalculating perplexity for {desc}...")
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    print(f"  Tokens: {seq_len}, Iterations: ~{(seq_len - max_length) // stride + 1}")
    
    nlls = []
    prev_end_loc = 0
    
    model.eval()
    
    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc=desc):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            outputs = model(input_ids, labels=target_ids, use_cache=False)
            neg_log_likelihood = outputs.loss * trg_len
            
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    perplexity_value = ppl.item()
    
    print(f"  Perplexity: {perplexity_value:.2f}")
    
    return perplexity_value


def prepare_dataset(tokenizer, seq_len=2048, cache_dir=None):
    """
    Prepare WikiText-2 train/validation datasets for causal LM training.
    """
    print("\n" + "="*70)
    print("Loading WikiText-2 Dataset")
    print("="*70)
    
    # Load WikiText-2
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    # Concatenate all text for each split
    train_text = "\n\n".join([text for text in dataset["train"]["text"] if text.strip()])
    val_text = "\n\n".join([text for text in dataset["validation"]["text"] if text.strip()])
    
    print(f"\nTokenizing train text...")
    train_encodings = tokenizer(train_text, return_tensors="pt", truncation=False)
    train_input_ids = train_encodings["input_ids"][0]
    
    print(f"Tokenizing validation text...")
    val_encodings = tokenizer(val_text, return_tensors="pt", truncation=False)
    val_input_ids = val_encodings["input_ids"][0]
    
    print(f"\nTrain tokens: {len(train_input_ids)}")
    print(f"Validation tokens: {len(val_input_ids)}")
    
    # Split into chunks of seq_len
    def create_chunks(input_ids, seq_len):
        chunks = []
        for i in range(0, len(input_ids) - seq_len + 1, seq_len):
            chunk = input_ids[i:i + seq_len]
            chunks.append({
                "input_ids": chunk.tolist(),
                "labels": chunk.tolist()
            })
        return chunks
    
    print(f"\nCreating chunks of {seq_len} tokens...")
    train_chunks = create_chunks(train_input_ids, seq_len)
    val_chunks = create_chunks(val_input_ids, seq_len)
    
    # Convert to datasets
    train_dataset = Dataset.from_list(train_chunks)
    val_dataset = Dataset.from_list(val_chunks)
    
    print(f"\nFinal train blocks: {len(train_dataset)}")
    print(f"Final validation blocks: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Task-Adapted Clean Baseline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="saved_models/wikitext2_clean_adapted_fp16",
                        help="Output directory for task-adapted clean model")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum training steps (default: 200 for task adaptation)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device (default: 1 for OOM safety)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for training")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 training")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for datasets")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Task-Adapted Clean Baseline Training (Method A)")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"FP16: {args.fp16}")
    print("="*70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load original clean model
    print("\n" + "="*70)
    print("Loading Original Clean Model")
    print("="*70)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Use float32 for stable training
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager',
        _attn_implementation_internal='eager'
    )
    
    print(f"Model loaded on {model.device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Evaluate baseline (zero-shot) perplexity
    print("\n" + "="*70)
    print("Baseline (Zero-Shot) Perplexity Evaluation")
    print("="*70)
    
    # Load WikiText-2 test set
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=args.cache_dir)
    test_text = "\n\n".join([text for text in dataset["test"]["text"] if text.strip()])
    
    baseline_ppl = calculate_perplexity(model, tokenizer, test_text, desc="Baseline (Zero-Shot)")
    
    # Prepare training dataset
    train_dataset, val_dataset = prepare_dataset(tokenizer, seq_len=args.seq_len, cache_dir=args.cache_dir)
    
    # Training arguments
    print("\n" + "="*70)
    print("Configuring Training")
    print("="*70)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,  # Use float32 to avoid gradient scaling issues
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=False,  # Disable for faster training
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM (not masked LM)
    )
    
    # Trainer (standard - NO anchoring)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*70)
    print("Training Task-Adapted Clean Model")
    print("="*70)
    print("NOTE: This is standard LM training with NO weight anchoring")
    print("="*70)
    
    train_result = trainer.train()
    
    # Save final model
    print("\n" + "="*70)
    print("Saving Task-Adapted Clean Model")
    print("="*70)
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model saved to: {args.output_dir}")
    
    # Evaluate final perplexity
    print("\n" + "="*70)
    print("Final Perplexity Evaluation")
    print("="*70)
    
    final_ppl = calculate_perplexity(model, tokenizer, test_text, desc="Task-Adapted Clean")
    
    # Save training log
    training_log = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "training_args": {
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "fp16": args.fp16,
            "seq_len": args.seq_len
        },
        "perplexity": {
            "baseline_zero_shot": baseline_ppl,
            "task_adapted_final": final_ppl,
            "improvement": baseline_ppl - final_ppl,
            "improvement_pct": ((baseline_ppl - final_ppl) / baseline_ppl) * 100
        },
        "train_metrics": {
            "train_loss": train_result.metrics.get("train_loss"),
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second")
        },
        "timestamp": datetime.now().isoformat()
    }
    
    log_file = "task_adapted_clean_training_log.json"
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining log saved to: {log_file}")
    
    # Summary
    print("\n" + "="*70)
    print("TASK-ADAPTED CLEAN BASELINE TRAINING COMPLETE")
    print("="*70)
    print(f"Baseline (Zero-Shot) PPL: {baseline_ppl:.2f}")
    print(f"Task-Adapted PPL: {final_ppl:.2f}")
    print(f"Improvement: {baseline_ppl - final_ppl:.2f} ({((baseline_ppl - final_ppl) / baseline_ppl) * 100:.1f}%)")
    print(f"\nModel saved to: {args.output_dir}")
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Generate attacked models FROM this checkpoint:")
    print(f"   python weight_noise_from_adapted.py --base_model_path {args.output_dir}")
    print("\n2. Defend attacked models using this as anchor:")
    print(f"   python defense_anchor_finetune_adapted.py --clean_model_path {args.output_dir}")
    print("\n3. Evaluate with proper baseline comparison")
    print("="*70)


if __name__ == "__main__":
    main()
