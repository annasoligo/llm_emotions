#!/usr/bin/env python3
"""Merge LoRA adapter with base model and upload to HuggingFace."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-3-27b-it")
    parser.add_argument("--adapter", default="annasoli/gemma3-27b-dpo-calm-full")
    parser.add_argument("--output-repo", default="annasoli/gemma3-27b-dpo-calm-full-merged")
    parser.add_argument("--local-dir", default=None, help="Optional local save path")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to HuggingFace")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading processor (for image processor config): {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    if args.local_dir:
        print(f"Saving merged model locally to: {args.local_dir}")
        model.save_pretrained(args.local_dir)
        tokenizer.save_pretrained(args.local_dir)
        processor.save_pretrained(args.local_dir)

    if not args.no_upload:
        print(f"Pushing merged model to: {args.output_repo}")
        model.push_to_hub(args.output_repo, private=False)
        tokenizer.push_to_hub(args.output_repo, private=False)
        processor.push_to_hub(args.output_repo, private=False)

    print("Done!")

if __name__ == "__main__":
    main()
