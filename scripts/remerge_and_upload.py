#!/usr/bin/env python3
"""Re-merge LoRA adapters with base model including vision encoder, then upload to HF."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel
from huggingface_hub import HfApi
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output-name", required=True, help="HF repo name (e.g., gemma3-27b-teacher-mode-merged)")
    parser.add_argument("--base-model", default="google/gemma-3-27b-it", help="Base model")
    parser.add_argument("--local-output", default=None, help="Local output path (optional)")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading tokenizer and processor from base model")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    # Determine output path
    output_path = args.local_output or f"/tmp/merged_{args.output_name}"
    print(f"Saving merged model to: {output_path}")

    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)  # This saves preprocessor_config.json

    print(f"Uploading to HuggingFace: annasoli/{args.output_name}")
    api = HfApi()
    api.create_repo(f"annasoli/{args.output_name}", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=output_path,
        repo_id=f"annasoli/{args.output_name}",
        commit_message=f"Upload merged model with vision encoder"
    )

    print(f"Done! Model available at: https://huggingface.co/annasoli/{args.output_name}")

    # Cleanup if using temp path
    if not args.local_output and os.path.exists(output_path):
        print(f"Cleaning up temp directory: {output_path}")
        shutil.rmtree(output_path)

if __name__ == "__main__":
    main()
