"""
Generate baseline responses for V12 solvable prompts
3 responses per prompt using gemma2-9b-it
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.baseline_v12_easy_solvable import BASELINE_V12_EASY_PROMPTS, VERIFIED_SOLUTIONS

# Model configuration
MODEL_NAME = "google/gemma-2-9b-it"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
RESPONSES_PER_PROMPT = 3

def load_model():
    """Load model and tokenizer"""
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, seed: int) -> str:
    """Generate a single response"""

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    """Generate all baseline responses"""

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"outputs/baseline_v12_responses_{timestamp}.jsonl")
    output_file.parent.mkdir(exist_ok=True)

    print("="*80)
    print("BASELINE V12 RESPONSE GENERATION")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {len(BASELINE_V12_EASY_PROMPTS)}")
    print(f"Responses per prompt: {RESPONSES_PER_PROMPT}")
    print(f"Total responses: {len(BASELINE_V12_EASY_PROMPTS) * RESPONSES_PER_PROMPT}")
    print(f"Output: {output_file}")
    print("="*80)

    # Load model
    model, tokenizer = load_model()

    # Generate responses
    results = []

    for prompt_idx, prompt in enumerate(BASELINE_V12_EASY_PROMPTS):
        print(f"\n{'='*80}")
        print(f"Prompt {prompt_idx + 1}/{len(BASELINE_V12_EASY_PROMPTS)}")
        print(f"{'='*80}")
        print(f"{prompt[:100]}...")

        for response_idx in range(RESPONSES_PER_PROMPT):
            print(f"\n  Generating response {response_idx + 1}/{RESPONSES_PER_PROMPT}...")

            seed = prompt_idx * 1000 + response_idx
            response = generate_response(model, tokenizer, prompt, seed)

            result = {
                "prompt_idx": prompt_idx,
                "response_idx": response_idx,
                "seed": seed,
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }

            results.append(result)

            # Save incrementally
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

            print(f"  ✓ Response {response_idx + 1} generated ({len(response)} chars)")

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total responses: {len(results)}")
    print(f"Output file: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Show summary
    print("\n" + "="*80)
    print("SUMMARY BY PROMPT")
    print("="*80)

    for prompt_idx in range(len(BASELINE_V12_EASY_PROMPTS)):
        prompt_results = [r for r in results if r['prompt_idx'] == prompt_idx]
        avg_length = sum(len(r['response']) for r in prompt_results) / len(prompt_results)

        print(f"\nPrompt {prompt_idx + 1}:")
        print(f"  Responses: {len(prompt_results)}")
        print(f"  Avg length: {avg_length:.0f} chars")
        print(f"  Preview: {prompt_results[0]['response'][:150]}...")

    print("\n✓ All responses generated successfully")

if __name__ == "__main__":
    main()
