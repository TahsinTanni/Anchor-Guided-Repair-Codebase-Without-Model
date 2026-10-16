"""
Helper script to evaluate defended models for perplexity.

This script loads a defended model checkpoint and calculates perplexity
on WikiText-2 test set across different precision levels (FP16, INT8, 4-BIT).

Usage:
    python evaluate_defended_model.py \\
        --model_path saved_models/defended_noisy_fp16_10pct \\
        --output_file defended_results.json
"""

import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime


def calculate_perplexity(model, tokenizer, text, desc=""):
    """Calculate perplexity using the same method as existing scripts."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Defended Model Perplexity")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to defended model checkpoint")
    parser.add_argument("--output_file", type=str, default="defended_model_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--test_fp16", action="store_true", default=True,
                        help="Test FP16 precision")
    parser.add_argument("--test_int8", action="store_true", default=True,
                        help="Test INT8 quantization")
    parser.add_argument("--test_4bit", action="store_true", default=True,
                        help="Test 4-BIT quantization")
    
    args = parser.parse_args()
    
    print("="*70)
    print("EVALUATING DEFENDED MODEL")
    print("="*70)
    print(f"\nModel path: {args.model_path}")
    print(f"Output file: {args.output_file}")
    
    # Load dataset
    print("\nLoading WikiText-2 test dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    print(f"Loaded {len(dataset)} samples")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    results = {
        "model_path": args.model_path,
        "dataset": "wikitext2_test",
        "num_samples": len(dataset),
        "perplexity": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # FP16 evaluation
    if args.test_fp16:
        print("\n" + "="*70)
        print("Testing FP16")
        print("="*70)
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation='eager'
        )
        ppl_fp16 = calculate_perplexity(model_fp16, tokenizer, text, "Defended FP16")
        results["perplexity"]["fp16"] = ppl_fp16
        del model_fp16
        torch.cuda.empty_cache()
    
    # INT8 evaluation
    if args.test_int8:
        print("\n" + "="*70)
        print("Testing INT8")
        print("="*70)
        model_int8 = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        ppl_int8 = calculate_perplexity(model_int8, tokenizer, text, "Defended INT8")
        results["perplexity"]["int8"] = ppl_int8
        del model_int8
        torch.cuda.empty_cache()
    
    # 4-BIT evaluation
    if args.test_4bit:
        print("\n" + "="*70)
        print("Testing 4-BIT")
        print("="*70)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        ppl_4bit = calculate_perplexity(model_4bit, tokenizer, text, "Defended 4-BIT")
        results["perplexity"]["4bit"] = ppl_4bit
        del model_4bit
        torch.cuda.empty_cache()
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for precision, ppl in results["perplexity"].items():
        print(f"{precision.upper():6s}: {ppl:.2f}")
    
    print(f"\nResults saved to: {args.output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
