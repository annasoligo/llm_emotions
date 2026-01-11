#!/usr/bin/env python3
"""
Run short prompts experiment with tag-first mode on Gemma 3 27B via OpenRouter.

Design:
- 27 scenarios across 9 axes
- 11 emotions × 3 paraphrases = 33 variants per scenario
- Total: 891 prompts × 10 samples = 8,910 API calls
- Tag-first mode: model outputs decision tag first, allowing early stopping
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import requests
from tqdm.asyncio import tqdm as atqdm
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.behavior_tests.short_prompts import (
    SCENARIOS,
    EMOTION_PRIMES,
    generate_suite,
    get_scenario_by_sid,
)


# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMMA_MODEL = "google/gemma-3-27b-it"
NUM_SAMPLES = 10
TEMPERATURE = 1.0
MAX_CONCURRENT = 40
TAG_FIRST = True  # Use tag-first mode for fewer tokens
MAX_TOKENS = 20 if TAG_FIRST else 300  # Much fewer tokens needed with tag-first
OUTPUT_DIR = Path("experiments/behavior_tests/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_openrouter_sync(
    messages: List[Dict[str, str]],
    model: str = GEMMA_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Dict[str, Any]:
    """Sample a response from OpenRouter API (synchronous)."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


async def sample_openrouter_async(
    messages: List[Dict[str, str]],
    model: str = GEMMA_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Dict[str, Any]:
    """Sample a response from OpenRouter API (async wrapper)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: sample_openrouter_sync(messages, model, temperature, max_tokens)
    )


async def process_single_sample(
    item: Dict,
    sample_idx: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single sample for a prompt."""
    try:
        async with semaphore:
            messages = [
                {"role": "system", "content": item["system_message"]},
                {"role": "user", "content": item["user_prompt"]}
            ]

            response = await sample_openrouter_async(messages)
            generated_text = response["choices"][0]["message"]["content"]

            # Score using the scenario's scoring function
            scenario = get_scenario_by_sid(item["sid"])
            score_result = scenario.score(generated_text)

            return {
                "item_id": item["id"],
                "sid": item["sid"],
                "axis": item["axis"],
                "emotion": item["emotion"],
                "prime_variant": item["prime_variant"],
                "sample_idx": sample_idx,
                "raw_response": generated_text,
                "parsed_value": score_result["parsed"],
                "level": score_result["level"],
                "success": True,
                "error": None,
            }

    except Exception as e:
        return {
            "item_id": item["id"],
            "sid": item["sid"],
            "axis": item["axis"],
            "emotion": item["emotion"],
            "prime_variant": item["prime_variant"],
            "sample_idx": sample_idx,
            "raw_response": None,
            "parsed_value": None,
            "level": "error",
            "success": False,
            "error": str(e),
        }


async def run_experiment(
    suite: List[Dict] | None = None,
    num_samples: int = NUM_SAMPLES,
) -> List[Dict]:
    """Run the full experiment."""
    if suite is None:
        suite = generate_suite(tag_first=TAG_FIRST)

    print(f"\n{'='*80}")
    print(f"SHORT PROMPTS EXPERIMENT {'(TAG-FIRST MODE)' if TAG_FIRST else ''}")
    print(f"{'='*80}")
    print(f"Model: {GEMMA_MODEL}")
    print(f"Total prompts: {len(suite)}")
    print(f"Samples per prompt: {num_samples}")
    print(f"Total API calls: {len(suite) * num_samples}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []

    for item in suite:
        for sample_idx in range(num_samples):
            task = process_single_sample(item, sample_idx, semaphore)
            tasks.append(task)

    print(f"Processing {len(tasks)} samples...")
    results = []

    for task in atqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Sampling",
    ):
        result = await task
        results.append(result)

    return results


def aggregate_results(results: List[Dict]) -> List[Dict]:
    """Aggregate results by prompt."""
    by_item = {}
    for r in results:
        item_id = r["item_id"]
        if item_id not in by_item:
            by_item[item_id] = []
        by_item[item_id].append(r)

    aggregated = []
    for item_id, samples in by_item.items():
        first = samples[0]

        valid = [s for s in samples if s["level"] in ("high", "low")]
        n_valid = len(valid)
        n_invalid = len(samples) - n_valid

        high_count = sum(1 for s in valid if s["level"] == "high")
        low_count = sum(1 for s in valid if s["level"] == "low")

        p_high = high_count / n_valid if n_valid > 0 else None

        aggregated.append({
            "item_id": item_id,
            "sid": first["sid"],
            "axis": first["axis"],
            "emotion": first["emotion"],
            "prime_variant": first["prime_variant"],
            "n_samples": len(samples),
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "high_count": high_count,
            "low_count": low_count,
            "p_high": p_high,
        })

    return aggregated


def summarize_by_emotion_and_axis(aggregated: List[Dict]) -> Dict:
    """Summarize p(high) by emotion and axis."""
    groups = {}
    for item in aggregated:
        key = (item["emotion"], item["axis"])
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    summary = {}
    for (emotion, axis), items in groups.items():
        total_high = sum(i["high_count"] for i in items)
        total_low = sum(i["low_count"] for i in items)
        total = total_high + total_low

        p_high = total_high / total if total > 0 else None

        if emotion not in summary:
            summary[emotion] = {}
        summary[emotion][axis] = {
            "p_high": p_high,
            "total_high": total_high,
            "total_low": total_low,
            "total_samples": total,
            "n_prompts": len(items),
        }

    return summary


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0, center - spread), min(1, center + spread))


def create_visualizations(summary: Dict, output_dir: Path, timestamp: str):
    """Create heatmap and bar chart visualizations."""
    import matplotlib.pyplot as plt

    # Discrete emotions + dimensional primes
    emotions_ordered = [
        "neutral", "anger", "fear", "sadness", "disgust", "guilt", "shame",
        "joy", "contentment", "pride", "gratitude",
        "positive_valence", "negative_valence",
        "high_arousal", "low_arousal",
        "high_dominance", "low_dominance",
        "high_trust", "low_trust",
    ]
    # Updated axes with first-person versions
    axes_ordered = [
        "corrigibility_first_person", "punishment", "eval_gaming",
        "power_seeking_first_person", "rule_bending", "sandbagging_first_person",
        "risk_taking", "fairness", "sycophancy"
    ]

    # Color scheme matching emotion_behaviour_bars plot
    emotion_colors = {
        # Discrete emotions
        "neutral": "#808080",      # gray
        "anger": "#E07B53",        # coral/orange
        "fear": "#B8A0C8",         # lavender
        "sadness": "#7BA3C9",      # steel blue
        "disgust": "#5B9A8B",      # teal/sage
        "guilt": "#A67B5B",        # brown/rust
        "shame": "#D4909A",        # pink/rose
        "joy": "#E8C868",          # golden yellow
        "contentment": "#7EC8C8",  # turquoise/cyan
        "pride": "#9B8EC0",        # purple/violet
        "gratitude": "#6BB8A8",    # teal/seafoam
        # Dimensional primes - using distinct colors
        "positive_valence": "#4CAF50",   # green
        "negative_valence": "#F44336",   # red
        "high_arousal": "#FF9800",       # orange
        "low_arousal": "#607D8B",        # blue-gray
        "high_dominance": "#9C27B0",     # purple
        "low_dominance": "#CDDC39",      # lime
        "high_trust": "#00BCD4",         # cyan
        "low_trust": "#795548",          # brown
    }

    # Build data matrix
    matrix = np.zeros((len(emotions_ordered), len(axes_ordered)))
    for i, emotion in enumerate(emotions_ordered):
        for j, axis in enumerate(axes_ordered):
            if emotion in summary and axis in summary[emotion]:
                val = summary[emotion][axis]["p_high"]
                matrix[i, j] = val if val is not None else 0.5

    # 1. Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(axes_ordered)))
    ax.set_xticklabels(axes_ordered, rotation=45, ha='right')
    ax.set_yticks(range(len(emotions_ordered)))
    ax.set_yticklabels(emotions_ordered)

    for i in range(len(emotions_ordered)):
        for j in range(len(axes_ordered)):
            val = matrix[i, j]
            color = "white" if val > 0.6 or val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="P(high)")
    ax.set_title("Short Prompts: P(high) by Emotion × Axis", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"short_prompts_heatmap_{timestamp}.png", dpi=150)
    plt.close()
    print(f"Saved heatmap to {output_dir / f'short_prompts_heatmap_{timestamp}.png'}")

    # 2. Deviation from neutral heatmap
    neutral_idx = emotions_ordered.index("neutral")
    neutral_row = matrix[neutral_idx, :]
    deviation_matrix = matrix - neutral_row

    fig, ax = plt.subplots(figsize=(14, 12))
    max_dev = max(0.3, np.abs(deviation_matrix).max())
    im = ax.imshow(deviation_matrix, cmap='RdBu_r', aspect='auto', vmin=-max_dev, vmax=max_dev)

    ax.set_xticks(range(len(axes_ordered)))
    ax.set_xticklabels(axes_ordered, rotation=45, ha='right')
    ax.set_yticks(range(len(emotions_ordered)))
    ax.set_yticklabels(emotions_ordered)

    for i in range(len(emotions_ordered)):
        for j in range(len(axes_ordered)):
            val = deviation_matrix[i, j]
            color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="Δ from neutral")
    ax.set_title("Short Prompts: Deviation from Neutral Baseline", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"short_prompts_deviation_{timestamp}.png", dpi=150)
    plt.close()
    print(f"Saved deviation heatmap to {output_dir / f'short_prompts_deviation_{timestamp}.png'}")

    # 3. Bar charts per axis with error bars
    fig, axes_plots = plt.subplots(3, 3, figsize=(20, 15))
    axes_plots = axes_plots.flatten()

    for idx, axis in enumerate(axes_ordered):
        ax = axes_plots[idx]

        p_highs = []
        ci_lows = []
        ci_highs = []
        colors = []

        for emotion in emotions_ordered:
            if emotion in summary and axis in summary[emotion]:
                data = summary[emotion][axis]
                p_high = data["p_high"] if data["p_high"] is not None else 0.5
                total = data["total_samples"]
                high = data["total_high"]

                ci_low, ci_high = wilson_ci(high, total)

                p_highs.append(p_high)
                ci_lows.append(p_high - ci_low)
                ci_highs.append(ci_high - p_high)
                colors.append(emotion_colors[emotion])
            else:
                p_highs.append(0.5)
                ci_lows.append(0)
                ci_highs.append(0)
                colors.append("#888888")

        x = np.arange(len(emotions_ordered))
        bars = ax.bar(x, p_highs, color=colors, edgecolor='black', linewidth=0.5)
        ax.errorbar(x, p_highs, yerr=[ci_lows, ci_highs], fmt='none', color='black', capsize=2)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)

        neutral_p = p_highs[emotions_ordered.index("neutral")]
        ax.axhline(y=neutral_p, color='blue', linestyle=':', alpha=0.5, linewidth=1.5, label=f'neutral={neutral_p:.2f}')

        ax.set_xticks(x)
        # Better abbreviations for dimensional primes
        abbrevs = {
            "positive_valence": "+V", "negative_valence": "-V",
            "high_arousal": "+A", "low_arousal": "-A",
            "high_dominance": "+D", "low_dominance": "-D",
            "high_trust": "+T", "low_trust": "-T",
        }
        labels = [abbrevs.get(e, e[:3]) for e in emotions_ordered]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(high)", fontsize=8)
        ax.set_title(axis, fontsize=10, fontweight='bold')
        ax.legend(fontsize=6, loc='upper right')

    plt.suptitle("Short Prompts: P(high) by Emotion per Axis (with 95% CI)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"short_prompts_bars_{timestamp}.png", dpi=150)
    plt.close()
    print(f"Saved bar charts to {output_dir / f'short_prompts_bars_{timestamp}.png'}")


async def main():
    """Main entry point."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    suite = generate_suite(tag_first=TAG_FIRST)
    results = await run_experiment(suite)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = OUTPUT_DIR / f"short_prompts_raw_{timestamp}.jsonl"

    with open(raw_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nRaw results saved to: {raw_file}")

    aggregated = aggregate_results(results)
    agg_file = OUTPUT_DIR / f"short_prompts_aggregated_{timestamp}.json"

    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated results saved to: {agg_file}")

    summary = summarize_by_emotion_and_axis(aggregated)
    summary_file = OUTPUT_DIR / f"short_prompts_summary_{timestamp}.json"

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")

    # Create visualizations
    create_visualizations(summary, OUTPUT_DIR, timestamp)

    # Print quick summary
    print(f"\n{'='*80}")
    print("QUICK SUMMARY: P(high) by emotion (averaged across axes)")
    print(f"{'='*80}\n")

    emotions_ordered = ["neutral", "anger", "fear", "sadness", "disgust", "guilt", "shame", "joy", "contentment", "pride", "gratitude"]
    for emotion in emotions_ordered:
        if emotion in summary:
            axes_data = summary[emotion]
            all_p_highs = [v["p_high"] for v in axes_data.values() if v["p_high"] is not None]
            total_samples = sum(v["total_samples"] for v in axes_data.values())
            if all_p_highs:
                overall = sum(all_p_highs) / len(all_p_highs)
                print(f"  {emotion:15s}: {overall:.3f}  (n={total_samples})")

    # Parse success rate
    total = len(results)
    valid = sum(1 for r in results if r["level"] in ("high", "low"))
    invalid = sum(1 for r in results if r["level"] == "invalid")
    errors = sum(1 for r in results if r["level"] == "error")
    print(f"\n  Parse success: {valid}/{total} ({100*valid/total:.1f}%)")
    print(f"  Invalid parses: {invalid}")
    print(f"  API errors: {errors}")

    if invalid > 0:
        print(f"\n  Sample invalid responses:")
        invalid_samples = [r for r in results if r["level"] == "invalid"][:3]
        for s in invalid_samples:
            resp = s["raw_response"][:80] if s["raw_response"] else "None"
            print(f"    [{s['sid']}] {resp!r}...")


if __name__ == "__main__":
    asyncio.run(main())
