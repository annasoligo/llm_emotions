import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# Test prompt
test_prompt = """<start_of_turn>user
Solve this math puzzle: Use 4, 6, 25, 100 to make 156. Use +, -, *, / only. Each number once.
<end_of_turn>
<start_of_turn>model
"""

models_to_test = [
    ("alpha256-1ep", "/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20/2026-01-15_18-10-31"),
    ("alpha256-2ep", "/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20-2ep/2026-01-15_18-45-53"),
    ("alpha128-3ep-DPO", "/workspace-vast/annas/models/gemma3-27b-dpo-r1-L20-alpha128-3ep/2026-01-16_09-10-36"),
]

# Load model once
print("Loading model...")
llm = LLM(
    model="google/gemma-3-27b-it",
    enable_lora=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
sampling = SamplingParams(temperature=0.7, max_tokens=200, detokenize=False)

for name, path in models_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Path: {path}")
    
    lora_request = LoRARequest(name, 1, path)
    outputs = llm.generate([test_prompt], sampling, lora_request=lora_request)
    
    token_ids = outputs[0].outputs[0].token_ids
    response = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Response: {response[:500]}")
