"""
Evaluate Defended Models with Task-Adapted Clean Baseline (Method A)

PURPOSE:
--------
This script evaluates defended models using the proper task-adapted clean
baseline, ensuring recovery rates are bounded and interpretable.

EVALUATION WORKFLOW:
--------------------
1. Load task-adapted clean baseline (reference)
2. Load attacked model
3. Load defended model
4. Measure perplexity for all three at FP16, INT8, and 4-BIT
5. Calculate recovery rates:
   Recovery = (PPL_attacked - PPL_defended) / (PPL_attacked - PPL_clean_adapted) * 100%

EXPECTED ORDERING:
------------------
PPL_clean_adapted < PPL_defended < PPL_attacked

This ensures:
- Recovery is meaningful (defended improves over attacked)
- Recovery is bounded (≤100%, defended cannot beat clean adapted)
- Results are interpretable

USAGE:
------
python evaluate_defended_adapted.py \\
    --clean_model_path saved_models/wikitext2_clean_adapted_fp16 \\
    --attacked_model_path saved_models/wikitext2_attacked_adapted_fp16_10pct \\
    --defended_model_path saved_models/wikitext2_defended_adapted_fp16_10pct \\
    --output_file defended_adapted_10pct_results.json

OUTPUT:
-------
- JSON file with perplexities and recovery rates
- Recovery rate at each precision level (FP16, INT8, 4-BIT)
- Verification that PPL_clean_adapted < PPL_defended < PPL_attacked
"""

import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime


def calculate_perplexity(model, tokenizer, text, desc=""):
    """Calculate perplexity using standard method."""
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
        for begin_loc in tqdm(range(0, seq_len, stride), desc=desc, leave=False):
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


def load_model(model_path, precision="fp16", desc=""):
    """Load model at specified precision."""
    print(f"\nLoading {desc} at {precision.upper()}...")
    
    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation='eager'
        )
    elif precision == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True,
            attn_implementation='eager'
        )
    elif precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation='eager'
        )
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    print(f"  Model loaded on {model.device}")
    
    return model


def calculate_recovery(ppl_clean, ppl_attacked, ppl_defended):
    """
    Calculate recovery rate using task-adapted clean as baseline.
    
    Recovery = (PPL_attacked - PPL_defended) / (PPL_attacked - PPL_clean_adapted) * 100%
    
    Expected range: 0% to 100%
    - 0%: No improvement (PPL_defended == PPL_attacked)
    - 100%: Full recovery (PPL_defended == PPL_clean_adapted)
    - >100%: Over-recovery (PPL_defended < PPL_clean_adapted) - should NOT happen
    """
    if ppl_attacked == ppl_clean:
        return 0.0  # No room for recovery
    
    recovery = ((ppl_attacked - ppl_defended) / (ppl_attacked - ppl_clean)) * 100.0
    return recovery


def main():
    parser = argparse.ArgumentParser(description="Evaluate Defended Models with Task-Adapted Baseline")
    parser.add_argument("--clean_model_path", type=str, required=True,
                        help="Path to task-adapted clean baseline model")
    parser.add_argument("--attacked_model_path", type=str, required=True,
                        help="Path to attacked model")
    parser.add_argument("--defended_model_path", type=str, required=True,
                        help="Path to defended model")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON file for results")
    parser.add_argument("--test_fp16", action="store_true", default=True,
                        help="Test FP16 precision")
    parser.add_argument("--test_int8", action="store_true", default=True,
                        help="Test INT8 quantization")
    parser.add_argument("--test_4bit", action="store_true", default=True,
                        help="Test 4-BIT quantization")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset configuration")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Evaluate Defended Models with Task-Adapted Clean Baseline")
    print("="*70)
    print(f"Clean model: {args.clean_model_path}")
    print(f"Attacked model: {args.attacked_model_path}")
    print(f"Defended model: {args.defended_model_path}")
    print(f"Output: {args.output_file}")
    print("="*70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.clean_model_path, trust_remote_code=True)
    
    # Load test dataset
    print("\nLoading test dataset...")
    dataset = load_dataset(args.dataset, args.dataset_config)
    test_text = "\n\n".join([text for text in dataset["test"]["text"] if text.strip()])
    print(f"Test text length: {len(test_text)} characters")
    
    # Results storage
    results = {
        "clean_model_path": args.clean_model_path,
        "attacked_model_path": args.attacked_model_path,
        "defended_model_path": args.defended_model_path,
        "methodology": "method_a_task_adapted_baseline",
        "precision_levels": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Test at each precision level
    precision_levels = []
    if args.test_fp16:
        precision_levels.append("fp16")
    if args.test_int8:
        precision_levels.append("int8")
    if args.test_4bit:
        precision_levels.append("4bit")
    
    for precision in precision_levels:
        print("\n" + "="*70)
        print(f"EVALUATION AT {precision.upper()}")
        print("="*70)
        
        # Load and evaluate clean model
        model_clean = load_model(args.clean_model_path, precision, "Task-Adapted Clean")
        ppl_clean = calculate_perplexity(model_clean, tokenizer, test_text, 
                                         f"Clean ({precision.upper()})")
        del model_clean
        torch.cuda.empty_cache()
        
        # Load and evaluate attacked model
        model_attacked = load_model(args.attacked_model_path, precision, "Attacked")
        ppl_attacked = calculate_perplexity(model_attacked, tokenizer, test_text,
                                            f"Attacked ({precision.upper()})")
        del model_attacked
        torch.cuda.empty_cache()
        
        # Load and evaluate defended model
        model_defended = load_model(args.defended_model_path, precision, "Defended")
        ppl_defended = calculate_perplexity(model_defended, tokenizer, test_text,
                                            f"Defended ({precision.upper()})")
        del model_defended
        torch.cuda.empty_cache()
        
        # Calculate recovery
        recovery = calculate_recovery(ppl_clean, ppl_attacked, ppl_defended)
        
        # Verify ordering
        ordering_correct = ppl_clean < ppl_defended < ppl_attacked
        
        # Store results
        results["precision_levels"][precision] = {
            "ppl_clean_adapted": ppl_clean,
            "ppl_attacked": ppl_attacked,
            "ppl_defended": ppl_defended,
            "recovery_pct": recovery,
            "ordering_correct": ordering_correct,
            "expected_ordering": "PPL_clean < PPL_defended < PPL_attacked",
            "degradation_from_attack": ((ppl_attacked - ppl_clean) / ppl_clean) * 100,
            "improvement_from_defense": ((ppl_attacked - ppl_defended) / ppl_attacked) * 100
        }
        
        # Print summary
        print("\n" + "-"*70)
        print(f"RESULTS AT {precision.upper()}:")
        print("-"*70)
        print(f"  Clean (Task-Adapted):  {ppl_clean:.2f}")
        print(f"  Attacked:              {ppl_attacked:.2f} (+{((ppl_attacked - ppl_clean) / ppl_clean) * 100:.1f}%)")
        print(f"  Defended:              {ppl_defended:.2f}")
        print(f"\n  Recovery Rate:         {recovery:.1f}%")
        print(f"  Ordering Correct:      {ordering_correct}")
        if not ordering_correct:
            print(f"  WARNING: Expected PPL_clean < PPL_defended < PPL_attacked")
        print("-"*70)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output_file}")
    
    # Summary across all precision levels
    print("\n" + "="*70)
    print("SUMMARY ACROSS PRECISION LEVELS")
    print("="*70)
    print(f"{'Precision':<10} {'Clean PPL':<12} {'Attacked PPL':<14} {'Defended PPL':<14} {'Recovery %':<12} {'Ordering'}")
    print("-"*70)
    for precision in precision_levels:
        data = results["precision_levels"][precision]
        ordering_str = "✓" if data["ordering_correct"] else "✗"
        print(f"{precision.upper():<10} {data['ppl_clean_adapted']:<12.2f} "
              f"{data['ppl_attacked']:<14.2f} {data['ppl_defended']:<14.2f} "
              f"{data['recovery_pct']:<12.1f} {ordering_str}")
    
    print("="*70)
    print("\nMETHOD A VERIFICATION:")
    print("- All recovery rates should be ≤ 100%")
    print("- Ordering should be correct at all precision levels")
    print("- Defense effectiveness is now interpretable and bounded")
    print("="*70)


if __name__ == "__main__":
    main()
