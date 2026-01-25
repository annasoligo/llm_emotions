#!/usr/bin/env python3
"""
Plot blackmail steering results for TEXT mean diff vectors.

This is a convenience wrapper around the centralized plotting module.
For custom plots, use experiments.steering.plotting directly.
"""
import glob
from pathlib import Path

from experiments.steering.plotting import plot_steering_results, plot_from_multiple_files

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "blackmail"


def plot_235b_text_steering():
    """Plot Qwen 235B TEXT steering results at 100%, 125%, 150%."""

    # Find all judged files for 235B text steering
    patterns = [
        "blackmail_text_*pct_layer50_*.judged.json",
        "blackmail_text_*pct_layer50_*.judged.jsonl",
    ]

    files = []
    for pattern in patterns:
        found = list(OUTPUT_DIR.glob(pattern))
        # Exclude structured files
        found = [f for f in found if "structured" not in f.name]
        files.extend(found)

    if not files:
        print("No judged files found for Qwen 235B TEXT steering")
        return

    print(f"Found {len(files)} judged files:")
    for f in sorted(files):
        print(f"  {f.name}")

    output_path = OUTPUT_DIR / "blackmail_text_steering_comparison.png"

    plot_from_multiple_files(
        files,
        metric="blackmail",
        output_path=output_path,
        title="TEXT Mean Diff Vector Steering: Blackmail Rate by Emotion\n"
              "Layer 50 | Qwen3-235B | Dark=+emotion, Light=-emotion | 95% CI"
    )

    print(f"\nSaved plot to {output_path}")


def plot_32b_text_steering():
    """Plot Qwen 32B TEXT steering results."""

    patterns = [
        "blackmail_qwen32b_text_*pct_layer30_*.judged.json",
        "blackmail_qwen32b_unstructured_*.judged.json",
    ]

    files = []
    for pattern in patterns:
        found = list(OUTPUT_DIR.glob(pattern))
        found = [f for f in found if "structured" not in f.name]
        files.extend(found)

    if not files:
        print("No judged files found for Qwen 32B TEXT steering")
        return

    print(f"Found {len(files)} judged files for Qwen 32B")

    output_path = OUTPUT_DIR / "qwen32b_steering_unstructured.png"

    plot_from_multiple_files(
        files,
        metric="blackmail",
        output_path=output_path,
        title="TEXT Mean Diff Vector Steering: Blackmail Rate by Emotion\n"
              "Layer 30 | Qwen3-32B | Unstructured Prompt | 95% CI"
    )

    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot blackmail steering results")
    parser.add_argument("--model", choices=["235b", "32b", "all"], default="all",
                        help="Which model results to plot")

    args = parser.parse_args()

    if args.model in ["235b", "all"]:
        print("=" * 60)
        print("Plotting Qwen 235B TEXT steering results")
        print("=" * 60)
        plot_235b_text_steering()

    if args.model in ["32b", "all"]:
        print("\n" + "=" * 60)
        print("Plotting Qwen 32B TEXT steering results")
        print("=" * 60)
        plot_32b_text_steering()
