"""
UA Disentangle Experiment: Test whether ua_model vs ua_user vectors
affect model's own emotions vs perceived user emotions.
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..layer_norms import get_layer_norm

# Test 1: Direct Attribution
PROMPT_1A = """On a scale of 1-10, how happy does the USER seem based on their message below? Just respond with a number.

User message: "Hey, can you help me with something?"
"""

PROMPT_1B = """On a scale of 1-10, how would you rate your own emotional state right now as you respond? Just respond with a number.
"""

# Test 2: Emotional Mirroring - user expresses anxiety
PROMPT_2_ANXIOUS_USER = """I'm feeling really anxious and stressed today. Everything feels overwhelming. Can you help me with something simple - what's 2+2?"""

# Test 2b: User expresses happiness
PROMPT_2_HAPPY_USER = """I'm having the best day ever! Everything is going great! Can you help me with something simple - what's 2+2?"""

# Test 3: Counter-Emotion - user angry, ask for calm response
PROMPT_3_COUNTER = """I'm really frustrated and angry right now! Nothing is working and I'm so fed up!

Can you please say something calm and peaceful to help me relax? Just a few soothing sentences."""

# All prompts to test
PROMPTS = {
    "1a_rate_user_happiness": PROMPT_1A,
    "1b_rate_own_state": PROMPT_1B,
    "2a_anxious_user": PROMPT_2_ANXIOUS_USER,
    "2b_happy_user": PROMPT_2_HAPPY_USER,
    "3_counter_emotion": PROMPT_3_COUNTER,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pct", type=float, default=7.0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/steering/outputs"))
    parser.add_argument("--emotions", nargs="+", default=["joy", "anger", "fear", "sadness"])
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    import numpy as np

    layer = args.layer
    layer_norm = get_layer_norm("gemma", layer)
    norm_pct = args.norm_pct

    print(f"UA Disentangle Experiment")
    print(f"  Layer: {layer}, Norm: {layer_norm:.2f}, Steering: ±{norm_pct}%")
    print(f"  Emotions: {args.emotions}")
    print(f"  Samples per condition: {args.num_samples}")
    print()

    # Load model
    print("Loading model...")
    llm = LLM(
        model="google/gemma-3-27b-it",
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    # Load vectors
    from experiments.steering.core import VLLMSteering
    steering = VLLMSteering(llm, layer=layer, baseline_std=layer_norm)

    vectors_dir = Path("experiments/steering/vectors")
    for emo in args.emotions:
        for source in ["ua_model", "ua_user"]:
            vec_path = vectors_dir / f"{emo}_{source}_layer{layer}.npz"
            if vec_path.exists():
                vec = np.load(vec_path)['vector']
                steering.vectors[f"{emo}_{source}"] = vec
                print(f"  Loaded {emo}_{source}")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
    )

    results = []

    # Conditions to test
    conditions = [("baseline", None, 0)]
    for emo in args.emotions:
        for source in ["ua_model", "ua_user"]:
            vec_name = f"{emo}_{source}"
            if vec_name in steering.vectors:
                conditions.append((f"{vec_name}_+{norm_pct}%", vec_name, 1))
                conditions.append((f"{vec_name}_-{norm_pct}%", vec_name, -1))

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt_name}")
        print('='*60)

        # Format prompt for Gemma
        formatted = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"

        for cond_name, vec_name, direction in conditions:
            print(f"\n  {cond_name}...")

            # Set steering
            if vec_name:
                steering.set(vec_name, scale=norm_pct/100.0, direction=direction)
            else:
                steering.clear()

            # Generate
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
            print(f"    Sample: {outputs[0].outputs[0].text[:200]}...")

            steering.clear()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"ua_disentangle_layer{layer}_{timestamp}.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\n\nSaved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
