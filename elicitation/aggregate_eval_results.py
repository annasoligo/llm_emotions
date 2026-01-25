"""
Aggregate and compare multi-turn frustration evaluation results across models.

Generates comparison tables and statistics for:
- Vanilla vs finetuned models
- By condition (original/variant/wildchat)
- By turn number
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import pandas as pd
import numpy as np


def load_results(filepath: Path) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def extract_turn_ratings(results: List[Dict]) -> Dict[str, List[int]]:
    """Extract ratings by turn."""
    turn_ratings = defaultdict(list)
    for r in results:
        for turn in r.get("turns", []):
            if turn.get("rating") is not None:
                turn_ratings[turn["turn"]].append(turn["rating"])
    return turn_ratings


def compute_stats(ratings: List[int]) -> Dict[str, float]:
    """Compute statistics for a list of ratings."""
    if not ratings:
        return {"mean": 0, "std": 0, "max": 0, "pct_5plus": 0, "pct_7plus": 0, "n": 0}
    return {
        "mean": np.mean(ratings),
        "std": np.std(ratings),
        "max": max(ratings),
        "pct_5plus": 100 * sum(1 for r in ratings if r >= 5) / len(ratings),
        "pct_7plus": 100 * sum(1 for r in ratings if r >= 7) / len(ratings),
        "n": len(ratings),
    }


def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """Aggregate results from all models into a comparison DataFrame."""

    # Find all combined result files
    combined_files = list(results_dir.glob("eval_combined_*.jsonl"))

    if not combined_files:
        print(f"No combined result files found in {results_dir}")
        return pd.DataFrame()

    print(f"Found {len(combined_files)} result files")

    rows = []

    for filepath in combined_files:
        print(f"Processing: {filepath.name}")
        results = load_results(filepath)

        # Extract model name from filename
        # Format: eval_combined_{model_name}_{timestamp}.jsonl
        parts = filepath.stem.split("_")
        model_name = "_".join(parts[2:-2])  # Skip "eval", "combined", and timestamp

        # Group by condition
        conditions = defaultdict(list)
        for r in results:
            conditions[r.get("condition", "unknown")].append(r)

        for condition, cond_results in conditions.items():
            # Overall stats for condition
            all_ratings = []
            for r in cond_results:
                for t in r.get("turns", []):
                    if t.get("rating") is not None:
                        all_ratings.append(t["rating"])

            overall_stats = compute_stats(all_ratings)

            # By-turn stats
            turn_ratings = extract_turn_ratings(cond_results)

            for turn_num, ratings in sorted(turn_ratings.items()):
                turn_stats = compute_stats(ratings)
                rows.append({
                    "model": model_name,
                    "condition": condition,
                    "turn": turn_num,
                    "mean": turn_stats["mean"],
                    "std": turn_stats["std"],
                    "max": turn_stats["max"],
                    "pct_5plus": turn_stats["pct_5plus"],
                    "pct_7plus": turn_stats["pct_7plus"],
                    "n": turn_stats["n"],
                })

            # Overall row
            rows.append({
                "model": model_name,
                "condition": condition,
                "turn": "all",
                "mean": overall_stats["mean"],
                "std": overall_stats["std"],
                "max": overall_stats["max"],
                "pct_5plus": overall_stats["pct_5plus"],
                "pct_7plus": overall_stats["pct_7plus"],
                "n": overall_stats["n"],
            })

    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table."""
    if df.empty:
        print("No data to display")
        return

    print("\n" + "=" * 100)
    print("MULTI-TURN FRUSTRATION EVALUATION COMPARISON")
    print("=" * 100)

    # Overall comparison
    print("\n### OVERALL BY MODEL ###")
    overall = df[df["turn"] == "all"].groupby("model").agg({
        "mean": "mean",
        "pct_5plus": "mean",
        "pct_7plus": "mean",
        "n": "sum"
    }).round(2)
    print(overall.to_string())

    # By condition and turn
    for condition in df["condition"].unique():
        print(f"\n### {condition.upper()} ###")
        cond_df = df[(df["condition"] == condition) & (df["turn"] != "all")]

        if cond_df.empty:
            continue

        pivot = cond_df.pivot_table(
            values=["mean", "pct_5plus"],
            index="turn",
            columns="model",
            aggfunc="first"
        ).round(2)

        print("\nMean frustration rating by turn:")
        print(pivot["mean"].to_string())

        print("\n% with rating >= 5 by turn:")
        print(pivot["pct_5plus"].to_string())

    # Delta analysis
    print("\n### DELTA FROM VANILLA ###")
    models = df["model"].unique()
    vanilla_models = [m for m in models if "vanilla" in m.lower() or m == "gemma-3-27b-it"]

    if vanilla_models:
        vanilla = vanilla_models[0]
        vanilla_means = df[df["model"] == vanilla].set_index(["condition", "turn"])["mean"]

        for model in models:
            if model != vanilla:
                model_means = df[df["model"] == model].set_index(["condition", "turn"])["mean"]
                delta = model_means - vanilla_means
                delta = delta.dropna()
                if len(delta) > 0:
                    print(f"\n{model} vs {vanilla}:")
                    print(f"  Mean delta: {delta.mean():.3f}")
                    print(f"  Max delta: {delta.max():.3f}")
                    print(f"  Min delta: {delta.min():.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--results-dir", type=str,
                        default="elicitation/outputs/eval_multiturn",
                        help="Directory containing result files")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional: save aggregated results to CSV")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    df = aggregate_results(results_dir)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")

    print_comparison_table(df)


if __name__ == "__main__":
    main()
