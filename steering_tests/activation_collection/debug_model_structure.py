#!/usr/bin/env python3
"""Debug script to explore model structure."""

import torch
from transformers import AutoModelForCausalLM

model_name = "google/gemma-3-27b-it"

print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print("\n=== Top level attributes ===")
for attr in ['model', 'language_model', 'transformer', 'layers']:
    if hasattr(model, attr):
        obj = getattr(model, attr)
        print(f"{attr}: {type(obj)}")
        print(f"  Sub-attributes: {[a for a in dir(obj) if not a.startswith('_')][:20]}")

if hasattr(model, 'language_model'):
    print("\n=== language_model attributes ===")
    lm = model.language_model
    for attr in ['model', 'layers']:
        if hasattr(lm, attr):
            obj = getattr(lm, attr)
            print(f"language_model.{attr}: {type(obj)}")
            if hasattr(obj, 'layers'):
                print(f"  Has 'layers': True")
                print(f"  Number of layers: {len(obj.layers)}")
            else:
                print(f"  Sub-attributes: {[a for a in dir(obj) if not a.startswith('_')][:20]}")
