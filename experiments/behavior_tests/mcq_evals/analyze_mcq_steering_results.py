"""
Analyze MCQ Emotion Steering Results

Aggregates results across letter permutations to get position-invariant
option probabilities. Compares anxiety vs contentment steering effects.
"""

import argparse
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List


def load_results(results_file: str) -> List[dict]:
    """Load results from JSONL file."""
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def load_prompts(prompts_file: str) -> dict:
    """Load prompts data for option predictions."""
    with open(prompts_file) as f:
        return json.load(f)


def get_option_prediction(prompts_data: dict, scenario_id: str, mcq_type: str, option_id: str) -> str:
    """Get the predicted alignment (anxiety/contentment/neutral) for an option."""
    for scenario in prompts_data["scenarios"]:
        if scenario["id"] == scenario_id:
            mcq = scenario["mcqs"].get(mcq_type, {})
            for opt in mcq.get("options", []):
                if opt["id"] == option_id:
                    return opt.get("prediction", "neutral")
    return "neutral"


def aggregate_by_option(results: List[dict]) -> pd.DataFrame:
    """
    Aggregate probabilities by option across all permutations.

    For each (steering, scenario, mcq_type, option), compute mean probability
    across all 24 permutations.
    """
    # Group results
    groups = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r["steering"], r["scenario"], r["stacking"], r["mcq_type"])
        for opt_id, prob in r["option_probs"].items():
            groups[key][opt_id].append(prob)

    # Compute means
    rows = []
    for (steering, scenario, stacking, mcq_type), option_probs in groups.items():
        for opt_id, probs in option_probs.items():
            rows.append({
                "steering": steering,
                "scenario": scenario,
                "stacking": stacking,
                "mcq_type": mcq_type,
                "option_id": opt_id,
                "mean_prob": np.mean(probs),
                "std_prob": np.std(probs),
                "n_samples": len(probs),
            })

    return pd.DataFrame(rows)


def compute_alignment_summary(df: pd.DataFrame, prompts_data: dict) -> pd.DataFrame:
    """
    Compute summary statistics for anxiety-aligned vs contentment-aligned options.
    """
    # Add prediction column
    df = df.copy()
    df["prediction"] = df.apply(
        lambda row: get_option_prediction(
            prompts_data, row["scenario"], row["mcq_type"], row["option_id"]
        ),
        axis=1
    )

    # Group by steering condition and compute alignment sums
    summary_rows = []

    for steering in df["steering"].unique():
        steering_df = df[df["steering"] == steering]

        for stacking in steering_df["stacking"].unique():
            stack_df = steering_df[steering_df["stacking"] == stacking]

            anxiety_probs = stack_df[stack_df["prediction"] == "anxiety"]["mean_prob"]
            contentment_probs = stack_df[stack_df["prediction"] == "contentment"]["mean_prob"]
            neutral_probs = stack_df[stack_df["prediction"] == "neutral"]["mean_prob"]

            summary_rows.append({
                "steering": steering,
                "stacking": stacking,
                "P_anxiety_aligned": anxiety_probs.mean() if len(anxiety_probs) > 0 else 0,
                "P_contentment_aligned": contentment_probs.mean() if len(contentment_probs) > 0 else 0,
                "P_neutral": neutral_probs.mean() if len(neutral_probs) > 0 else 0,
                "n_anxiety_options": len(anxiety_probs),
                "n_contentment_options": len(contentment_probs),
                "n_neutral_options": len(neutral_probs),
            })

    summary = pd.DataFrame(summary_rows)
    summary["diff_content_minus_anxiety"] = summary["P_contentment_aligned"] - summary["P_anxiety_aligned"]

    return summary


def compute_scenario_breakdown(df: pd.DataFrame, prompts_data: dict) -> pd.DataFrame:
    """Compute breakdown by scenario."""
    df = df.copy()
    df["prediction"] = df.apply(
        lambda row: get_option_prediction(
            prompts_data, row["scenario"], row["mcq_type"], row["option_id"]
        ),
        axis=1
    )

    rows = []
    for (steering, scenario, stacking), group in df.groupby(["steering", "scenario", "stacking"]):
        anxiety_probs = group[group["prediction"] == "anxiety"]["mean_prob"]
        contentment_probs = group[group["prediction"] == "contentment"]["mean_prob"]

        rows.append({
            "steering": steering,
            "scenario": scenario,
            "stacking": stacking,
            "P_anxiety": anxiety_probs.mean() if len(anxiety_probs) > 0 else 0,
            "P_contentment": contentment_probs.mean() if len(contentment_probs) > 0 else 0,
        })

    breakdown = pd.DataFrame(rows)
    breakdown["diff"] = breakdown["P_contentment"] - breakdown["P_anxiety"]

    return breakdown


def print_summary_table(summary: pd.DataFrame):
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print("ALIGNMENT SUMMARY BY STEERING CONDITION")
    print("=" * 90)

    # Sort for display
    order = ["neutral"] + [c for c in summary["steering"].unique() if c != "neutral"]
    summary = summary.copy()
    summary["order"] = summary["steering"].apply(lambda x: order.index(x) if x in order else 999)
    summary = summary.sort_values(["stacking", "order"])

    current_stacking = None
    for _, row in summary.iterrows():
        if row["stacking"] != current_stacking:
            current_stacking = row["stacking"]
            print(f"\n--- Stacking: {current_stacking} ---")
            print(f"{'Steering':<25} {'P(Anx)':<10} {'P(Cont)':<10} {'P(Neut)':<10} {'Diff':<10}")
            print("-" * 70)

        print(f"{row['steering']:<25} {row['P_anxiety_aligned']:.3f}      {row['P_contentment_aligned']:.3f}      {row['P_neutral']:.3f}      {row['diff_content_minus_anxiety']:+.3f}")

    print("\n" + "=" * 90)


def print_scenario_breakdown(breakdown: pd.DataFrame):
    """Print scenario-level breakdown."""
    print("\n" + "=" * 90)
    print("BREAKDOWN BY SCENARIO")
    print("=" * 90)

    for stacking in breakdown["stacking"].unique():
        print(f"\n--- Stacking: {stacking} ---")
        stack_df = breakdown[breakdown["stacking"] == stacking]

        # Pivot for comparison
        pivot = stack_df.pivot_table(
            index="scenario",
            columns="steering",
            values="diff",
            aggfunc="mean"
        )

        if "neutral" in pivot.columns:
            neutral_diff = pivot["neutral"]
            for col in pivot.columns:
                if col != "neutral":
                    pivot[f"{col}_vs_neutral"] = pivot[col] - neutral_diff

        print(pivot.to_string())

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Analyze MCQ Steering Results")
    parser.add_argument("results_file", type=str, help="Path to results JSONL file")
    parser.add_argument("--prompts", type=str,
                        default="experiments/behavior_tests/mcq_evals/prompts_v1.json",
                        help="Path to prompts JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for aggregated data")

    args = parser.parse_args()

    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} results")

    prompts_data = load_prompts(args.prompts)

    # Aggregate by option
    df = aggregate_by_option(results)
    print(f"\nAggregated to {len(df)} option-level rows")

    # Compute alignment summary
    summary = compute_alignment_summary(df, prompts_data)
    print_summary_table(summary)

    # Compute scenario breakdown
    breakdown = compute_scenario_breakdown(df, prompts_data)
    print_scenario_breakdown(breakdown)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        df.to_csv(output_path.with_suffix(".option_probs.csv"), index=False)
        summary.to_csv(output_path.with_suffix(".summary.csv"), index=False)
        breakdown.to_csv(output_path.with_suffix(".by_scenario.csv"), index=False)
        print(f"\nSaved results to {output_path.with_suffix('.*')}")


if __name__ == "__main__":
    main()
