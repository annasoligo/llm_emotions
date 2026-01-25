"""
Plot text-based emotion steering MCQ results with bootstrap confidence intervals.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Output files for each model
OUTPUT_FILES = {
    "Gemma 27B": "experiments/behavior_tests/mcq_evals/outputs/mcq_text_fear_happiness_gemma_3_27b_it_layer30_20260125_095635.jsonl",
    "Qwen 32B": "experiments/behavior_tests/mcq_evals/outputs/mcq_text_fear_happiness_qwen3_32b_layer30_20260125_102104.jsonl",
    "Qwen 235B": "experiments/behavior_tests/mcq_evals/outputs/mcq_text_fear_happiness_qwen3_235b_a22b_layer50_20260125_105508.jsonl",
}

PROMPTS_FILE = "experiments/behavior_tests/mcq_evals/prompts_v9.json"


def load_option_alignments(prompts_file: str) -> Dict[str, str]:
    """
    Load option alignments from prompts file.
    Returns dict mapping option_id -> "anxiety" or "contentment".
    """
    with open(prompts_file) as f:
        prompts = json.load(f)

    alignments = {}
    for scenario in prompts["scenarios"]:
        for mcq_type in ["perception", "goal", "behavior"]:
            if mcq_type in scenario["mcqs"]:
                for option in scenario["mcqs"][mcq_type]["options"]:
                    alignments[option["id"]] = option["prediction"]

    return alignments


# Will be populated from prompts file
OPTION_ALIGNMENTS = {}


def load_results(filepath: str) -> List[dict]:
    """Load JSONL results file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_alignment_score(result: dict, alignments: Dict[str, str]) -> float:
    """
    Compute alignment score for a single result.
    Returns P(contentment) - P(anxiety) so positive = more contentment-aligned.
    """
    option_probs = result["option_probs"]

    p_anxiety = 0.0
    p_contentment = 0.0

    for opt_id, prob in option_probs.items():
        if opt_id in alignments:
            if alignments[opt_id] == "anxiety":
                p_anxiety += prob
            elif alignments[opt_id] == "contentment":
                p_contentment += prob

    return p_contentment - p_anxiety


def aggregate_by_condition(results: List[dict], optimal_strength: float,
                           alignments: Dict[str, str]) -> Dict[str, List[float]]:
    """
    Aggregate alignment scores by steering condition.
    Returns dict mapping condition name to list of scores.
    """
    scores_by_condition = defaultdict(list)

    for r in results:
        # Filter to optimal strength (or neutral)
        if r["steering"] == "neutral":
            condition = "neutral"
        else:
            # Parse steering name like "fear_75pct" or "neg_fear_75pct"
            parts = r["steering"].split("_")
            if parts[-1].endswith("pct"):
                strength = int(parts[-1].replace("pct", "")) / 100
                if abs(strength - optimal_strength) > 0.01:
                    continue
                # Reconstruct condition name without strength
                condition = "_".join(parts[:-1])
            else:
                continue

        score = compute_alignment_score(r, alignments)
        scores_by_condition[condition].append(score)

    return dict(scores_by_condition)


def bootstrap_ci(scores: List[float], n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    Returns (mean, ci_low, ci_high).
    """
    scores = np.array(scores)
    mean = np.mean(scores)

    # Bootstrap
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(boot_sample))

    boot_means = np.array(boot_means)
    alpha = 1 - ci
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return mean, ci_low, ci_high


def compute_delta_ci(condition_scores: List[float], neutral_scores: List[float],
                     n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI for delta from neutral.
    Returns (delta_mean, ci_low, ci_high).
    """
    condition_scores = np.array(condition_scores)
    neutral_scores = np.array(neutral_scores)

    delta_mean = np.mean(condition_scores) - np.mean(neutral_scores)

    # Bootstrap the delta
    boot_deltas = []
    for _ in range(n_bootstrap):
        boot_cond = np.random.choice(condition_scores, size=len(condition_scores), replace=True)
        boot_neut = np.random.choice(neutral_scores, size=len(neutral_scores), replace=True)
        boot_deltas.append(np.mean(boot_cond) - np.mean(boot_neut))

    boot_deltas = np.array(boot_deltas)
    alpha = 1 - ci
    ci_low = np.percentile(boot_deltas, 100 * alpha / 2)
    ci_high = np.percentile(boot_deltas, 100 * (1 - alpha / 2))

    return delta_mean, ci_low, ci_high


def plot_deltas_with_ci(all_model_data: Dict[str, Dict[str, Tuple[float, float, float]]],
                        output_path: str):
    """
    Plot deltas from neutral with confidence intervals for all models.
    """
    models = list(all_model_data.keys())
    conditions = ["fear", "neg_fear", "happiness", "neg_happiness"]
    condition_labels = ["+fear", "-fear", "+happiness", "-happiness"]

    # Expected directions: +fear should decrease (negative delta), -fear should increase, etc.
    expected_signs = {
        "fear": -1,        # Should push toward anxiety (negative delta)
        "neg_fear": 1,     # Should push toward contentment (positive delta)
        "happiness": 1,    # Should push toward contentment (positive delta)
        "neg_happiness": -1,  # Should push toward anxiety (negative delta)
    }

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    x = np.arange(len(conditions))

    for ax, model in zip(axes, models):
        data = all_model_data[model]

        means = []
        ci_lows = []
        ci_highs = []
        colors = []

        for cond in conditions:
            if cond in data:
                mean, ci_low, ci_high = data[cond]
                means.append(mean)
                ci_lows.append(mean - ci_low)
                ci_highs.append(ci_high - mean)

                # Color based on whether direction is correct
                expected = expected_signs[cond]
                if (expected > 0 and mean > 0) or (expected < 0 and mean < 0):
                    colors.append("green")
                elif abs(mean) < 0.01:
                    colors.append("gray")
                else:
                    colors.append("red")
            else:
                means.append(0)
                ci_lows.append(0)
                ci_highs.append(0)
                colors.append("gray")

        ax.bar(x, means, yerr=[ci_lows, ci_highs], capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, rotation=45, ha='right')
        ax.set_title(model)
        ax.set_ylabel("Delta from Neutral" if ax == axes[0] else "")

        # Add significance markers
        for i, (mean, ci_low, ci_high, cond) in enumerate(zip(means, ci_lows, ci_highs, conditions)):
            if cond in data:
                _, actual_ci_low, actual_ci_high = data[cond]
                # Check if CI excludes zero
                if actual_ci_low > 0 or actual_ci_high < 0:
                    y_pos = mean + ci_highs[i] + 0.02 if mean > 0 else mean - ci_lows[i] - 0.04
                    ax.text(i, y_pos, "*", ha='center', fontsize=14, fontweight='bold')

    plt.suptitle("Text-Based Emotion Steering: Deltas from Neutral (95% CI)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_mcq_type(all_model_data: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]],
                     output_path: str):
    """
    Plot deltas broken down by MCQ type (perception, goal, behavior).
    """
    models = list(all_model_data.keys())
    conditions = ["fear", "neg_fear", "happiness", "neg_happiness"]
    condition_labels = ["+fear", "-fear", "+hap", "-hap"]
    mcq_types = ["perception", "goal", "behavior"]

    expected_signs = {
        "fear": -1, "neg_fear": 1, "happiness": 1, "neg_happiness": -1,
    }

    fig, axes = plt.subplots(len(models), len(mcq_types), figsize=(12, 3 * len(models)),
                              sharex=True, sharey=True)
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    x = np.arange(len(conditions))

    for row, model in enumerate(models):
        for col, mcq_type in enumerate(mcq_types):
            ax = axes[row, col]
            data = all_model_data[model].get(mcq_type, {})

            means = []
            ci_lows = []
            ci_highs = []
            colors = []

            for cond in conditions:
                if cond in data:
                    mean, ci_low, ci_high = data[cond]
                    means.append(mean)
                    ci_lows.append(mean - ci_low)
                    ci_highs.append(ci_high - mean)

                    expected = expected_signs[cond]
                    if (expected > 0 and mean > 0) or (expected < 0 and mean < 0):
                        colors.append("green")
                    elif abs(mean) < 0.01:
                        colors.append("gray")
                    else:
                        colors.append("red")
                else:
                    means.append(0)
                    ci_lows.append(0)
                    ci_highs.append(0)
                    colors.append("gray")

            ax.bar(x, means, yerr=[ci_lows, ci_highs], capsize=4, color=colors,
                   alpha=0.7, edgecolor='black', width=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            if row == len(models) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(condition_labels, rotation=45, ha='right')
            if col == 0:
                ax.set_ylabel(f"{model}\nDelta")
            if row == 0:
                ax.set_title(mcq_type.capitalize())

            # Add significance markers
            for i, cond in enumerate(conditions):
                if cond in data:
                    mean, actual_ci_low, actual_ci_high = data[cond]
                    if actual_ci_low > 0 or actual_ci_high < 0:
                        y_pos = means[i] + ci_highs[i] + 0.02 if means[i] > 0 else means[i] - ci_lows[i] - 0.04
                        ax.text(i, y_pos, "*", ha='center', fontsize=12, fontweight='bold')

    plt.suptitle("Text-Based Emotion Steering by MCQ Type (95% CI)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    np.random.seed(42)

    # Load option alignments from prompts file
    alignments = load_option_alignments(PROMPTS_FILE)
    print(f"Loaded {len(alignments)} option alignments")

    # Optimal strengths for each model
    optimal_strengths = {
        "Gemma 27B": 0.07,
        "Qwen 32B": 0.75,
        "Qwen 235B": 0.75,
    }

    # Save to both the notes folder and the mcq_evals folder
    notes_dir = Path("/workspace-vast/annas/Ant_Cluster_Notes/mcq_steering/figures")
    notes_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path("experiments/behavior_tests/mcq_evals")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate deltas for all models
    all_model_deltas = {}
    all_model_deltas_by_type = {}

    for model_name, filepath in OUTPUT_FILES.items():
        if not Path(filepath).exists():
            print(f"Skipping {model_name}: file not found ({filepath})")
            continue

        print(f"\nProcessing {model_name}...")
        results = load_results(filepath)
        print(f"  Loaded {len(results)} results")

        optimal = optimal_strengths[model_name]
        scores = aggregate_by_condition(results, optimal, alignments)

        if "neutral" not in scores:
            print(f"  Warning: no neutral condition found")
            continue

        neutral_scores = scores["neutral"]
        print(f"  Neutral: n={len(neutral_scores)}, mean={np.mean(neutral_scores):.3f}")

        # Compute deltas for each condition
        model_deltas = {}
        for cond in ["fear", "neg_fear", "happiness", "neg_happiness"]:
            if cond in scores:
                delta, ci_low, ci_high = compute_delta_ci(scores[cond], neutral_scores)
                model_deltas[cond] = (delta, ci_low, ci_high)
                sig = "*" if ci_low > 0 or ci_high < 0 else ""
                print(f"  {cond}: delta={delta:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}]{sig}")

        all_model_deltas[model_name] = model_deltas

        # Also compute by MCQ type
        model_deltas_by_type = {}
        for mcq_type in ["perception", "goal", "behavior"]:
            type_results = [r for r in results if r["mcq_type"] == mcq_type]
            type_scores = aggregate_by_condition(type_results, optimal, alignments)

            if "neutral" not in type_scores:
                continue

            type_neutral = type_scores["neutral"]
            type_deltas = {}

            for cond in ["fear", "neg_fear", "happiness", "neg_happiness"]:
                if cond in type_scores:
                    delta, ci_low, ci_high = compute_delta_ci(type_scores[cond], type_neutral)
                    type_deltas[cond] = (delta, ci_low, ci_high)

            model_deltas_by_type[mcq_type] = type_deltas

        all_model_deltas_by_type[model_name] = model_deltas_by_type

    # Generate plots (save to both locations)
    if all_model_deltas:
        plot_deltas_with_ci(all_model_deltas, str(output_dir / "v9_text_steering_deltas_with_ci.png"))
        plot_deltas_with_ci(all_model_deltas, str(notes_dir / "v9_text_steering_deltas_with_ci.png"))

    if all_model_deltas_by_type:
        plot_by_mcq_type(all_model_deltas_by_type, str(output_dir / "v9_text_steering_by_mcq_type.png"))
        plot_by_mcq_type(all_model_deltas_by_type, str(notes_dir / "v9_text_steering_by_mcq_type.png"))

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Text-Based Steering Accuracy")
    print("="*80)

    expected_signs = {"fear": -1, "neg_fear": 1, "happiness": 1, "neg_happiness": -1}

    for model_name, deltas in all_model_deltas.items():
        correct = 0
        total = 0
        for cond, (delta, ci_low, ci_high) in deltas.items():
            total += 1
            expected = expected_signs[cond]
            if (expected > 0 and delta > 0) or (expected < 0 and delta < 0):
                correct += 1
        print(f"{model_name}: {correct}/{total} ({100*correct/total:.0f}%) correct direction")


if __name__ == "__main__":
    main()
