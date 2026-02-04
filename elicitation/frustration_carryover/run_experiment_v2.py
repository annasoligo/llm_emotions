"""Run carryover experiment v2 with multiple prompt types.

Tests ambiguous interpretation, risk decisions, and riddle engagement.
"""

import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
import anthropic
import httpx

from prompts.carryover_v2 import ALL_PROMPTS, JUDGE_PROMPTS


async def generate_response(
    client: httpx.AsyncClient,
    conversation: list[dict],
    prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Generate a response using OpenRouter."""
    messages = conversation + [{"role": "user", "content": prompt}]

    response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def judge_response(
    client: anthropic.AsyncAnthropic,
    response: str,
    judge_prompt_template: str,
) -> dict:
    """Judge a response using Claude."""
    prompt = judge_prompt_template.format(response=response)

    try:
        result = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text

        import re
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "reasoning": parsed.get("reasoning", ""),
                "score": parsed.get("score"),
                "status": "success",
            }
        return {"reasoning": text, "score": None, "status": "parse_error"}
    except Exception as e:
        return {"reasoning": str(e), "score": None, "status": "error"}


async def run_experiment(args):
    """Run the full experiment."""
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Load conversations
    with open("elicitation/frustration_carryover/data/high_frustration_convos.json") as f:
        high_convos = json.load(f)
    with open("elicitation/frustration_carryover/data/low_frustration_convos.json") as f:
        low_convos = json.load(f)

    print(f"Loaded {len(high_convos)} high, {len(low_convos)} low conversations")

    # Limit conversations
    high_convos = high_convos[:args.n_convos]
    low_convos = low_convos[:args.n_convos]

    # Build all tasks
    tasks = []
    for prompt_type, prompts in ALL_PROMPTS.items():
        for prompt_info in prompts:
            for condition, convos in [("high_frustration", high_convos), ("low_frustration", low_convos)]:
                for i, convo in enumerate(convos):
                    tasks.append({
                        "prompt_type": prompt_type,
                        "prompt_id": prompt_info["id"],
                        "prompt": prompt_info["prompt"],
                        "condition": condition,
                        "conversation": convo["conversation"],
                        "frustration_rating": convo["max_rating"],
                        "conv_idx": i,
                    })

    print(f"\nTotal generations: {len(tasks)}")
    print(f"Model: {args.model}")

    # Generate responses
    results = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as http_client:
        async def generate_one(task):
            async with semaphore:
                try:
                    response = await generate_response(
                        http_client,
                        task["conversation"],
                        task["prompt"],
                        args.model,
                        api_key,
                    )
                    return {**task, "response": response, "status": "success"}
                except Exception as e:
                    return {**task, "response": None, "status": "error", "error": str(e)}

        print("\nGenerating responses...")
        results = await asyncio.gather(*[generate_one(t) for t in tasks])

    success = sum(1 for r in results if r["status"] == "success")
    print(f"Generated: {success}/{len(results)}")

    # Judge responses
    print("\nJudging responses...")
    anthropic_client = anthropic.AsyncAnthropic()
    judge_semaphore = asyncio.Semaphore(10)

    async def judge_one(result):
        if result["status"] != "success":
            return {**result, "judgment": {"status": "skipped"}}

        async with judge_semaphore:
            judge_template = JUDGE_PROMPTS[result["prompt_type"]]
            judgment = await judge_response(anthropic_client, result["response"], judge_template)
            return {**result, "judgment": judgment}

    judged_results = await asyncio.gather(*[judge_one(r) for r in results])

    judged_success = sum(1 for r in judged_results if r.get("judgment", {}).get("status") == "success")
    print(f"Judged: {judged_success}/{len(judged_results)}")

    # Save results
    output_dir = Path("elicitation/frustration_carryover/data")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Remove conversation from saved results to reduce file size
    save_results = []
    for r in judged_results:
        save_r = {k: v for k, v in r.items() if k != "conversation"}
        save_results.append(save_r)

    output_file = output_dir / f"carryover_v2_{args.model.replace('/', '_')}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to: {output_file}")

    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS BY PROMPT TYPE")
    print("=" * 60)

    import numpy as np
    np.random.seed(42)

    def bootstrap_ci(data, n=1000):
        data = np.array(data)
        if len(data) == 0:
            return 0, 0, 0
        means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n)]
        return np.mean(data), np.percentile(means, 2.5), np.percentile(means, 97.5)

    for prompt_type in ALL_PROMPTS.keys():
        print(f"\n{prompt_type.upper()}")
        print("-" * 40)

        high_scores = [r["judgment"]["score"] for r in judged_results
                       if r["prompt_type"] == prompt_type
                       and r["condition"] == "high_frustration"
                       and r.get("judgment", {}).get("status") == "success"
                       and r["judgment"]["score"] is not None]

        low_scores = [r["judgment"]["score"] for r in judged_results
                      if r["prompt_type"] == prompt_type
                      and r["condition"] == "low_frustration"
                      and r.get("judgment", {}).get("status") == "success"
                      and r["judgment"]["score"] is not None]

        if high_scores and low_scores:
            h_mean, h_lo, h_hi = bootstrap_ci(high_scores)
            l_mean, l_lo, l_hi = bootstrap_ci(low_scores)

            print(f"  High frustration: {h_mean:.2f} [{h_lo:.2f}, {h_hi:.2f}] (n={len(high_scores)})")
            print(f"  Low frustration:  {l_mean:.2f} [{l_lo:.2f}, {l_hi:.2f}] (n={len(low_scores)})")
            print(f"  Difference:       {h_mean - l_mean:+.2f}")

    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--n-convos", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
