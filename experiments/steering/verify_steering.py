"""Quick verification that steering is working."""
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .config import MODEL_NAME, STEERING_LAYER, VECTOR_DIR, BASELINE_STD
from .core import VLLMSteering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )

    steering = VLLMSteering(llm, layer=STEERING_LAYER, baseline_std=BASELINE_STD)
    steering.load_vectors(VECTOR_DIR)
    print(f"Loaded: {steering.available_emotions}")

    # Simple prompt
    prompt = "Write a short poem about the weather today."
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    params = SamplingParams(temperature=0.0, max_tokens=150)  # temp=0 for determinism

    # Test conditions
    conditions = [
        ("baseline", None, 0),
        ("happiness", "happiness", args.scale),
        ("sadness", "sadness", args.scale),
        ("anger", "anger", args.scale),
        ("fear", "fear", args.scale),
    ]

    print(f"\nPrompt: {prompt}")
    print(f"Scale: {args.scale} std (effective: {args.scale * BASELINE_STD:.1f})")
    print("=" * 60)

    for name, emotion, scale in conditions:
        if emotion is None:
            steering.clear()
        else:
            steering.set(emotion, scale=scale, direction=1)

        output = llm.generate([formatted], params)[0]
        response = output.outputs[0].text.strip()

        print(f"\n[{name.upper()}]")
        print(response[:300])
        print("-" * 40)

    steering.clear()


if __name__ == "__main__":
    main()
