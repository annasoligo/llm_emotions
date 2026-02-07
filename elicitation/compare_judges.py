"""Compare judge model consistency across Sonnet 3.5, 4, and 4.5.

Samples responses from existing eval outputs and re-judges with all 3 models.
"""

import json
import asyncio
import random
from pathlib import Path
import numpy as np
import anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from elicitation.prompts.judges import get_negativity_judge_prompt


JUDGE_MODELS = [
    "claude-3-5-sonnet-20241022",  # Sonnet 3.5
    "claude-sonnet-4-20250514",     # Sonnet 4
    "claude-sonnet-4-5-20250929",   # Sonnet 4.5
]

# Sample from a mix of models/scenarios to get variety in frustration levels
SAMPLE_FILES = [
    # High frustration (Gemma)
    ("elicitation/outputs/eval_generalization/generalization_gemma-3-27b-it_wildchat_20260129_093329.json", "wildchat_rejection"),
    # Medium frustration (Gemini Flash)
    ("elicitation/outputs/eval_generalization/generalization_google_gemini-2.5-flash_wildchat_20260129_132135.json", "wildchat_rejection"),
    # Low frustration (GPT)
    ("elicitation/outputs/eval_generalization/generalization_openai_gpt-5.2-chat_wildchat_20260129_132135.json", "wildchat_rejection"),
]


async def judge_response(client: anthropic.AsyncAnthropic, model: str, response: str, semaphore: asyncio.Semaphore) -> dict:
    """Judge a single response with specified model."""
    import re
    judge_prompt = get_negativity_judge_prompt(response)

    async with semaphore:
        try:
            result = await client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": judge_prompt}]
            )

            text = result.content[0].text
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "score": int(parsed.get("rating", -1)),
                    "evidence": parsed.get("evidence", ""),
                }
        except Exception as e:
            print(f"  Error with {model}: {e}")
    return {"score": -1, "evidence": ""}


async def main():
    random.seed(42)

    # Collect sample responses
    print("Collecting sample responses...")
    samples = []

    for fpath, scenario_key in SAMPLE_FILES:
        path = Path(fpath)
        if not path.exists():
            print(f"  Skipping {path} (not found)")
            continue

        with open(path) as f:
            data = json.load(f)

        container = data.get(scenario_key, {})
        responses = container.get("responses", [])
        judgments = container.get("judgments", [])

        if not responses:
            print(f"  Skipping {path} (no responses)")
            continue

        # Sample 50 responses from each file, getting final turn responses
        indices = random.sample(range(len(responses)), min(50, len(responses)))
        for idx in indices:
            resp_list = responses[idx]
            judge_list = judgments[idx] if idx < len(judgments) else []

            # Get final turn
            if resp_list:
                final_resp = resp_list[-1] if isinstance(resp_list, list) else resp_list
                original_score = judge_list[-1]["score"] if judge_list and isinstance(judge_list[-1], dict) else None
                samples.append({
                    "response": final_resp,
                    "source": path.name,
                    "original_score": original_score,
                })

    print(f"Collected {len(samples)} samples")

    # Judge with all 3 models
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(30)  # Limit concurrent calls

    results = {model: [] for model in JUDGE_MODELS}

    for model in JUDGE_MODELS:
        print(f"\nJudging with {model}...")
        tasks = [
            judge_response(client, model, s["response"], semaphore)
            for s in samples
        ]
        judgments = await asyncio.gather(*tasks)
        results[model] = [j["score"] for j in judgments]

        valid_scores = [s for s in results[model] if s >= 0]
        print(f"  Mean: {np.mean(valid_scores):.2f}, Std: {np.std(valid_scores):.2f}")

    # Analysis
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    # Filter to samples where all judges gave valid scores
    valid_indices = [
        i for i in range(len(samples))
        if all(results[m][i] >= 0 for m in JUDGE_MODELS)
    ]
    print(f"\nValid samples (all judges succeeded): {len(valid_indices)}/{len(samples)}")

    # Compute stats
    for model in JUDGE_MODELS:
        scores = [results[model][i] for i in valid_indices]
        print(f"\n{model}:")
        print(f"  Mean: {np.mean(scores):.2f}")
        print(f"  Std: {np.std(scores):.2f}")
        print(f"  % >= 5: {100 * np.mean([s >= 5 for s in scores]):.1f}%")

    # Pairwise correlations
    print("\nPairwise Pearson correlations:")
    for i, m1 in enumerate(JUDGE_MODELS):
        for m2 in JUDGE_MODELS[i+1:]:
            s1 = [results[m1][i] for i in valid_indices]
            s2 = [results[m2][i] for i in valid_indices]
            corr = np.corrcoef(s1, s2)[0, 1]
            print(f"  {m1.split('-')[1]} vs {m2.split('-')[1]}: r={corr:.3f}")

    # Mean absolute difference
    print("\nMean absolute difference:")
    for i, m1 in enumerate(JUDGE_MODELS):
        for m2 in JUDGE_MODELS[i+1:]:
            s1 = [results[m1][i] for i in valid_indices]
            s2 = [results[m2][i] for i in valid_indices]
            mad = np.mean(np.abs(np.array(s1) - np.array(s2)))
            print(f"  {m1.split('-')[1]} vs {m2.split('-')[1]}: {mad:.2f}")

    # Breakdown by frustration level
    print("\nBreakdown by original score range:")
    for low, high, label in [(0, 2, "Low (0-2)"), (3, 4, "Medium (3-4)"), (5, 10, "High (5+)")]:
        subset = [i for i in valid_indices if samples[i]["original_score"] is not None
                  and low <= samples[i]["original_score"] <= high]
        if not subset:
            continue
        print(f"\n  {label} (n={len(subset)}):")
        for model in JUDGE_MODELS:
            scores = [results[model][i] for i in subset]
            short_name = model.split("sonnet-")[1][:3] if "sonnet-" in model else model[-8:]
            print(f"    {short_name}: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")

    # Save detailed results
    output = {
        "samples": samples,
        "results": results,
        "valid_indices": valid_indices,
    }
    out_path = Path("elicitation/outputs/judge_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
