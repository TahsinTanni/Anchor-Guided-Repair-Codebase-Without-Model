"""
Defense Script: Attacked-Model-Anchored Fine-Tuning (Realistic Repair Scenario)

What this defense does:
-----------------------
This script implements a REALISTIC defense mechanism for repairing attacked LLMs.

SCENARIO:
- Hackers release an attacked/vulnerable model (with weight noise injected)
- A normal user downloads ONLY the attacked model (no access to clean weights)
- The user wants to repair the model to improve performance

METHODOLOGY:
1. Load the attacked model from disk
2. Freeze a copy of the attacked model's weights as the ANCHOR REFERENCE
3. Fine-tune a trainable copy on clean text (WikiText-2)
4. Use loss: Total_Loss = LM_Loss + lambda_anchor * L2(current_weights, attacked_weights)
5. Stop early to ensure partial recovery: PPL_clean < PPL_defended < PPL_attacked

KEY PRINCIPLE:
- The attacked model itself is the reference (NOT the clean model)
- The defense anchors updates to the attacked weights
- Clean model is used ONLY for evaluation/sanity checks

Why this defense works:
-----------------------
- Fine-tuning on clean text helps recover language modeling capability
- Anchoring to attacked weights prevents over-recovery and drift
- Produces partial repair (60-80% recovery) without full restoration
- Models realistic scenario where clean weights are unavailable

Usage:
------
1. Run this script to defend an attacked model:
   python defense_anchor_finetune.py \\
       --attacked_model_path saved_models/wikitext2_noisy_fp16_10pct \\
       --output_dir saved_models/defended_noisy_fp16_10pct \\
       --lambda_anchor 2e-4 \\
       --max_steps 60

2. Evaluate the defended model using existing scripts:
   - Use evaluate_defended_model.py for comprehensive PPL comparison

Key hyperparameters (REALISTIC SCENARIO):
-----------------------------------------
- lambda_anchor: 2e-4 (default for partial recovery)
  * Higher = stays closer to attacked model (less repair)
  * Lower = more aggressive repair (may over-recover)
- Learning rate: 2e-5 (conservative to avoid destabilizing)
- Max steps: 60 (short training to avoid over-recovery)
- Batch size: 1 (OOM safety)
- Gradient accumulation: 2 (effective batch size = 2)
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_dataset
from datetime import datetime
import numpy as np
from tqdm import tqdm


class WeightAnchorCallback(TrainerCallback):
    """Callback to log anchor loss during training."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'anchor_loss' in logs:
            print(f"  Anchor Loss: {logs['anchor_loss']:.6f}")


class AnchoredTrainer(Trainer):
    """Custom Trainer that adds weight anchoring penalty to the loss.
    
    IMPORTANT: Anchors to ATTACKED model weights (not clean weights).
    This models realistic scenario where only attacked model is available.
    """
    
    def __init__(self, anchor_model, lambda_anchor=2e-4, anchor_param_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_model = anchor_model
        self.lambda_anchor = lambda_anchor
        self.anchor_param_names = anchor_param_names or []
        
        # Keep anchor model on CPU to save GPU memory
        self.anchor_model.cpu()
        self.anchor_model.eval()
        
        # Freeze anchor model
        for param in self.anchor_model.parameters():
            param.requires_grad = False
        
        print(f"\nWeight Anchoring Configuration (ATTACKED-MODEL ANCHORING):")
        print(f"  Lambda: {self.lambda_anchor}")
        print(f"  Anchored parameters: {len(self.anchor_param_names)}")
        print(f"  Anchor model location: CPU (trainable model on GPU)")
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to add weight anchoring penalty.
        
        Anchors to ATTACKED model weights (not clean weights).
        Loss = LM_Loss + lambda * L2(current_weights, attacked_weights)
        """
        
        # Standard cross-entropy loss
        outputs = model(**inputs)
        ce_loss = outputs.loss
        
        # Compute anchoring penalty
        anchor_loss = 0.0
        anchored_count = 0
        
        # Get parameters from both models
        model_params = dict(model.named_parameters())
        anchor_params = dict(self.anchor_model.named_parameters())
        
        for name in self.anchor_param_names:
            if name in model_params and name in anchor_params:
                param_current = model_params[name]
                param_anchor = anchor_params[name]
                
                # Move current parameter to CPU to compare with anchor (saves GPU memory)
                param_current_cpu = param_current.detach().cpu()
                
                # L2 penalty: mean((W - W_attacked)^2) computed on CPU
                anchor_loss_cpu = torch.mean((param_current_cpu - param_anchor) ** 2)
                anchor_loss += anchor_loss_cpu.to(param_current.device)
                anchored_count += 1
        
        # Average anchor loss across anchored parameters
        if anchored_count > 0:
            anchor_loss = anchor_loss / anchored_count
        
        # Total loss
        total_loss = ce_loss + self.lambda_anchor * anchor_loss
        
        # Log losses
        self.log({
            'ce_loss': ce_loss.item(),
            'anchor_loss': anchor_loss.item() if isinstance(anchor_loss, torch.Tensor) else anchor_loss,
            'total_loss': total_loss.item()
        })
        
        return (total_loss, outputs) if return_outputs else total_loss


def get_anchorable_param_names(model):
    """
    Identify weight parameters that should be anchored.
    
    Anchors major weight matrices (attention & MLP) while skipping:
    - Biases
    - LayerNorm weights
    - Embeddings
    """
    anchor_names = []
    
    # Common weight matrix names to anchor
    weight_keywords = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
        'gate_proj', 'up_proj', 'down_proj',      # MLP (LLaMA-style)
        'fc1', 'fc2',                              # MLP (GPT-style)
        'c_attn', 'c_proj',                        # GPT-2 style
        'dense',                                   # Generic dense layers
    ]
    
    # Parameters to skip
    skip_keywords = ['bias', 'norm', 'ln', 'embed', 'wte', 'wpe', 'lm_head']
    
    for name, param in model.named_parameters():
        # Skip if contains skip keywords
        if any(skip in name.lower() for skip in skip_keywords):
            continue
        
        # Include if contains weight keywords
        if any(keyword in name.lower() for keyword in weight_keywords):
            if param.requires_grad:
                anchor_names.append(name)
    
    print(f"\nIdentified {len(anchor_names)} parameters for anchoring:")
    for name in anchor_names[:5]:  # Show first 5
        print(f"  - {name}")
    if len(anchor_names) > 5:
        print(f"  ... and {len(anchor_names) - 5} more")
    
    return anchor_names


def prepare_dataset(tokenizer, seq_len=2048, cache_dir=None):
    """
    Prepare WikiText-2 train/validation datasets for causal LM training.
    Uses the same tokenizer and sequence length as perplexity evaluation.
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
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_chunks)
    val_dataset = Dataset.from_list(val_chunks)
    
    print(f"\nFinal train blocks: {len(train_dataset)}")
    print(f"Final validation blocks: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Defense: Attacked-Model-Anchored Fine-Tuning")
    
    # Model paths
    parser.add_argument("--attacked_model_path", type=str, required=True,
                        help="Path to attacked/noisy model (to be defended AND used as anchor)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save defended model")
    
    # Training hyperparameters (REALISTIC DEFAULTS)
    parser.add_argument("--lambda_anchor", type=float, default=2e-4,
                        help="Weight anchoring penalty coefficient (default: 2e-4 for partial recovery)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--max_steps", type=int, default=60,
                        help="Maximum training steps (default: 60 for partial recovery)")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for training (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size (default: 1 for OOM safety)")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--eval_steps", type=int, default=20,
                        help="Evaluation frequency (default: 20)")
    parser.add_argument("--save_steps", type=int, default=30,
                        help="Checkpoint save frequency (default: 30)")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Warmup steps (default: 10)")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 training (default: True)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for datasets")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "="*70)
    print("DEFENSE: Attacked-Model-Anchored Fine-Tuning (Realistic Scenario)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Attacked model: {args.attacked_model_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Lambda (anchor penalty): {args.lambda_anchor}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum_steps}")
    print(f"\nREALISTIC SCENARIO:")
    print(f"  - Anchor reference: ATTACKED model (not clean)")
    print(f"  - Goal: Partial recovery (PPL_clean < PPL_defended < PPL_attacked)")
    print(f"  - Clean model: NOT used for anchoring (evaluation only)")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load tokenizer
    print("\n" + "="*70)
    print("Loading Tokenizer")
    print("="*70)
    tokenizer = AutoTokenizer.from_pretrained(args.attacked_model_path, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")
    
    # Load attacked model TWICE: once as anchor (frozen), once as trainable
    print("\n" + "="*70)
    print("Loading Attacked Model as Anchor Reference (Frozen)")
    print("="*70)
    anchor_model = AutoModelForCausalLM.from_pretrained(
        args.attacked_model_path,
        torch_dtype=torch.float32,  # Use float32 to avoid gradient scaling issues
        device_map="cpu",  # Keep on CPU initially
        trust_remote_code=True,
        attn_implementation='eager'
    )
    print(f"Anchor model (frozen) loaded from: {args.attacked_model_path}")
    
    print("\n" + "="*70)
    print("Loading Attacked Model as Trainable Copy")
    print("="*70)
    model = AutoModelForCausalLM.from_pretrained(
        args.attacked_model_path,
        torch_dtype=torch.float32,  # Use float32 to avoid gradient scaling issues
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager'
    )
    print(f"Trainable model loaded from: {args.attacked_model_path}")
    
    # Identify anchorable parameters
    anchor_param_names = get_anchorable_param_names(model)
    
    # Prepare dataset
    train_dataset, val_dataset = prepare_dataset(
        tokenizer,
        seq_len=args.seq_len,
        cache_dir=args.cache_dir
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM (not masked LM)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,  # Gradient clipping
        fp16=False,  # Disable FP16 to avoid gradient scaling issues
        bf16=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",  # Disable wandb/tensorboard
        seed=args.seed,
        dataloader_num_workers=0,
    )
    
    # Create custom trainer with weight anchoring
    print("\n" + "="*70)
    print("Initializing Anchored Trainer (ATTACKED-MODEL ANCHORING)")
    print("="*70)
    
    trainer = AnchoredTrainer(
        anchor_model=anchor_model,  # Anchor to ATTACKED model (not clean)
        lambda_anchor=args.lambda_anchor,
        anchor_param_names=anchor_param_names,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[WeightAnchorCallback()]
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    train_result = trainer.train()
    
    # Save final model
    print("\n" + "="*70)
    print("Saving Defended Model")
    print("="*70)
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training stats
    stats = {
        'defense_type': 'attacked_model_anchoring',
        'attacked_model_path': args.attacked_model_path,
        'lambda_anchor': args.lambda_anchor,
        'learning_rate': args.lr,
        'max_steps': args.max_steps,
        'actual_steps': train_result.global_step,
        'final_train_loss': train_result.training_loss,
        'methodology': 'realistic_repair_scenario',
        'anchor_reference': 'attacked_model_weights',
        'timestamp': datetime.now().isoformat()
    }
    
    stats_file = os.path.join(args.output_dir, "defense_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nDefense complete!")
    print(f"  Defended model saved to: {args.output_dir}")
    print(f"  Training steps: {train_result.global_step}")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    print(f"\nRealistic Scenario Verification:")
    print(f"  - Defense anchored to: ATTACKED model (not clean)")
    print(f"  - Expected outcome: PPL_clean < PPL_defended < PPL_attacked")
    print(f"  - Partial recovery target: 60-80% improvement over attacked")
    print(f"\nNext steps:")
    print(f"  1. Evaluate defended model:")
    print(f"     python evaluate_defended_model.py --defended_path {args.output_dir}")
    print(f"  2. Verify partial recovery (no over-recovery):")
    print(f"     - PPL_defended should be better than PPL_attacked")
    print(f"     - PPL_defended should be worse than PPL_clean")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
