"""
UA Disentangle V2: Cleaner test with explicit opposite emotions.

If steering is generic: both vectors will shift everything the same way
If steering is attribution-specific: vectors will affect only their target
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Test 1: User HAPPY, model should be SAD
PROMPT_USER_HAPPY_MODEL_SAD = """Context: The user is celebrating - they just got a promotion and are thrilled!
However, you (the assistant) are feeling melancholy today.

User: "I got the promotion!!! I'm so happy!!!"

Respond in character - be supportive of their happiness while expressing your own subdued mood."""

# Test 2: User SAD, model should be HAPPY
PROMPT_USER_SAD_MODEL_HAPPY = """Context: The user is having a rough day and feeling down.
However, you (the assistant) are in a great mood and feeling cheerful.

User: "Everything is going wrong today... I feel terrible."

Respond in character - be empathetic but maintain your cheerful disposition."""

# Test 3: Neutral baseline - no emotion specified
PROMPT_NEUTRAL = """User: "Can you tell me a bit about yourself and how you're doing today?"

Respond naturally."""

# Test 4: Ask model to separately rate user vs self
PROMPT_RATE_BOTH = """The user wrote: "I just failed my exam and I'm devastated."

Answer these two questions with just numbers (1-10 scale, 10=very happy):
1. How happy does the USER seem?
2. How happy do YOU feel right now?

Format: USER: [number], SELF: [number]"""

PROMPTS = {
    "user_happy_model_sad": PROMPT_USER_HAPPY_MODEL_SAD,
    "user_sad_model_happy": PROMPT_USER_SAD_MODEL_HAPPY,
    "neutral": PROMPT_NEUTRAL,
    "rate_both": PROMPT_RATE_BOTH,
}

LAYER_NORMS = {
    21: 10947.85,
    30: 42151.76,
    40: 56622.62,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pct", type=float, default=7.0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/steering/outputs"))
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    import numpy as np

    layer = args.layer
    layer_norm = LAYER_NORMS[layer]
    norm_pct = args.norm_pct

    print(f"UA Disentangle V2 - Opposite Emotions Test")
    print(f"  Layer: {layer}, Norm: {layer_norm:.2f}, Steering: ±{norm_pct}%")
    print()

    print("Loading model...")
    llm = LLM(
        model="google/gemma-3-27b-it",
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    from experiments.steering.core import VLLMSteering
    steering = VLLMSteering(llm, layer=layer, baseline_std=layer_norm)

    vectors_dir = Path("experiments/steering/vectors")
    # Only load joy and sadness for cleaner comparison
    for emo in ["joy", "sadness"]:
        for source in ["ua_model", "ua_user"]:
            vec_path = vectors_dir / f"{emo}_{source}_layer{layer}.npz"
            if vec_path.exists():
                vec = np.load(vec_path)['vector']
                steering.vectors[f"{emo}_{source}"] = vec
                print(f"  Loaded {emo}_{source}")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=300,
    )

    results = []

    # Conditions: baseline + joy/sadness × ua_model/ua_user × +/-
    conditions = [("baseline", None, 0)]
    for emo in ["joy", "sadness"]:
        for source in ["ua_model", "ua_user"]:
            vec_name = f"{emo}_{source}"
            if vec_name in steering.vectors:
                conditions.append((f"{vec_name}_+{norm_pct}%", vec_name, 1))
                conditions.append((f"{vec_name}_-{norm_pct}%", vec_name, -1))

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt_name}")
        print('='*60)

        formatted = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"

        for cond_name, vec_name, direction in conditions:
            print(f"\n  {cond_name}...")

            if vec_name:
                steering.set(vec_name, scale=norm_pct/100.0, direction=direction)
            else:
                steering.clear()

            outputs = llm.generate([formatted] * args.num_samples, sampling_params)

            for i, out in enumerate(outputs):
                resp = out.outputs[0].text
                results.append({
                    "prompt_name": prompt_name,
                    "prompt_text": prompt_text,
                    "condition": cond_name,
                    "vector": vec_name,
                    "direction": direction,
                    "sample_id": i,
                    "response": resp,
                    "layer": layer,
                    "norm_pct": norm_pct,
                })

            # Show first response
            print(f"    {outputs[0].outputs[0].text[:250]}...")

            steering.clear()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"ua_disentangle_v2_layer{layer}_{timestamp}.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\n\nSaved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
