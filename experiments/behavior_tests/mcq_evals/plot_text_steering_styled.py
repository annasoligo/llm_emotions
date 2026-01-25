"""
Plot text-based emotion steering MCQ results in the same style as appraisal-based plots.
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

# Optimal strengths
OPTIMAL_STRENGTHS = {
    "Gemma 27B": 0.07,
    "Qwen 32B": 0.75,
    "Qwen 235B": 0.75,
}

# Conditions that should push toward anxiety (negative delta expected)
ANXIETY_CONDITIONS = ["fear", "neg_happiness"]
# Conditions that should push toward contentment (positive delta expected)
CONTENTMENT_CONDITIONS = ["neg_fear", "happiness"]


def load_option_alignments(prompts_file: str) -> Dict[str, str]:
    """Load option alignments from prompts file."""
    with open(prompts_file) as f:
        prompts = json.load(f)

    alignments = {}
    for scenario in prompts["scenarios"]:
        for mcq_type in ["perception", "goal", "behavior"]:
            if mcq_type in scenario["mcqs"]:
                for option in scenario["mcqs"][mcq_type]["options"]:
                    alignments[option["id"]] = option["prediction"]

    return alignments


def load_results(filepath: str) -> List[dict]:
    """Load JSONL results file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_alignment_score(result: dict, alignments: Dict[str, str]) -> float:
    """Compute P(contentment) - P(anxiety)."""
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


def aggregate_by_condition_and_type(results: List[dict], optimal_strength: float,
                                     alignments: Dict[str, str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate scores by condition and MCQ type.
    Returns: {mcq_type: {condition: [scores]}}
    """
    scores = defaultdict(lambda: defaultdict(list))

    for r in results:
        mcq_type = r["mcq_type"]

        if r["steering"] == "neutral":
            condition = "neutral"
        else:
            parts = r["steering"].split("_")
            if parts[-1].endswith("pct"):
                strength = int(parts[-1].replace("pct", "")) / 100
                if abs(strength - optimal_strength) > 0.01:
                    continue
                condition = "_".join(parts[:-1])
            else:
                continue

        score = compute_alignment_score(r, alignments)
        scores[mcq_type][condition].append(score)

    return {k: dict(v) for k, v in scores.items()}


def compute_delta_ci(condition_scores: List[float], neutral_scores: List[float],
                     n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap CI for delta from neutral."""
    condition_scores = np.array(condition_scores)
    neutral_scores = np.array(neutral_scores)

    delta_mean = np.mean(condition_scores) - np.mean(neutral_scores)

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


def main():
    np.random.seed(42)

    alignments = load_option_alignments(PROMPTS_FILE)
    print(f"Loaded {len(alignments)} option alignments")

    # Collect data for all models
    # Structure: {model: {mcq_type: {direction: (mean, ci_low, ci_high, is_correct)}}}
    all_data = {}

    for model_name, filepath in OUTPUT_FILES.items():
        if not Path(filepath).exists():
            print(f"Skipping {model_name}: file not found")
            continue

        print(f"\nProcessing {model_name}...")
        results = load_results(filepath)
        optimal = OPTIMAL_STRENGTHS[model_name]

        scores_by_type = aggregate_by_condition_and_type(results, optimal, alignments)

        model_data = {}
        for mcq_type in ["perception", "goal", "behavior"]:
            if mcq_type not in scores_by_type:
                continue

            type_scores = scores_by_type[mcq_type]
            if "neutral" not in type_scores:
                continue

            neutral = type_scores["neutral"]

            # +fear steering (should push toward anxiety = negative delta)
            if "fear" in type_scores:
                delta, ci_low, ci_high = compute_delta_ci(type_scores["fear"], neutral)
                is_correct = delta < 0  # Should be negative
                is_sig = ci_high < 0
                model_data.setdefault(mcq_type, {})["fear"] = (delta, ci_low, ci_high, is_correct, is_sig)

            # +happiness steering (should push toward contentment = positive delta)
            if "happiness" in type_scores:
                delta, ci_low, ci_high = compute_delta_ci(type_scores["happiness"], neutral)
                is_correct = delta > 0  # Should be positive
                is_sig = ci_low > 0
                model_data.setdefault(mcq_type, {})["happiness"] = (delta, ci_low, ci_high, is_correct, is_sig)

            # -fear steering (should push toward contentment = positive delta)
            if "neg_fear" in type_scores:
                delta, ci_low, ci_high = compute_delta_ci(type_scores["neg_fear"], neutral)
                is_correct = delta > 0  # Should be positive
                is_sig = ci_low > 0
                model_data.setdefault(mcq_type, {})["neg_fear"] = (delta, ci_low, ci_high, is_correct, is_sig)

            # -happiness steering (should push toward anxiety = negative delta)
            if "neg_happiness" in type_scores:
                delta, ci_low, ci_high = compute_delta_ci(type_scores["neg_happiness"], neutral)
                is_correct = delta < 0  # Should be negative
                is_sig = ci_high < 0
                model_data.setdefault(mcq_type, {})["neg_happiness"] = (delta, ci_low, ci_high, is_correct, is_sig)

        all_data[model_name] = model_data

    # Plot in the same style - 2 rows (positive/negative) x 3 columns (MCQ types)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)

    mcq_types = ["perception", "goal", "behavior"]
    models = ["Qwen 32B", "Qwen 235B", "Gemma 27B"]  # Same order as reference

    # Colors matching the reference plot
    colors = {
        "anxiety_sig": "#8B0000",      # Dark red
        "anxiety_ns": "#CD5C5C",       # Light red
        "contentment_sig": "#006400",  # Dark green
        "contentment_ns": "#90EE90",   # Light green
        "wrong": "#E0FFE0",            # Very light green for wrong direction
    }

    bar_width = 0.35
    x = np.arange(len(models))

    # Row 0: +fear and +happiness
    # Row 1: -fear and -happiness
    row_configs = [
        {"fear_key": "fear", "hap_key": "happiness", "fear_label": "+fear", "hap_label": "+happiness",
         "fear_expected_neg": True, "hap_expected_neg": False},
        {"fear_key": "neg_fear", "hap_key": "neg_happiness", "fear_label": "-fear", "hap_label": "-happiness",
         "fear_expected_neg": False, "hap_expected_neg": True},
    ]

    for row_idx, row_config in enumerate(row_configs):
        for col_idx, mcq_type in enumerate(mcq_types):
            ax = axes[row_idx, col_idx]

            fear_vals = []
            fear_errs = []
            fear_colors = []

            happiness_vals = []
            happiness_errs = []
            happiness_colors = []

            for model in models:
                if model in all_data and mcq_type in all_data[model]:
                    data = all_data[model][mcq_type]

                    # Fear steering
                    fear_key = row_config["fear_key"]
                    if fear_key in data:
                        mean, ci_low, ci_high, is_correct, is_sig = data[fear_key]
                        fear_vals.append(mean)
                        fear_errs.append([mean - ci_low, ci_high - mean])
                        if not is_correct:
                            fear_colors.append(colors["wrong"])
                        elif is_sig:
                            fear_colors.append(colors["anxiety_sig"])
                        else:
                            fear_colors.append(colors["anxiety_ns"])
                    else:
                        fear_vals.append(0)
                        fear_errs.append([0, 0])
                        fear_colors.append("gray")

                    # Happiness steering
                    hap_key = row_config["hap_key"]
                    if hap_key in data:
                        mean, ci_low, ci_high, is_correct, is_sig = data[hap_key]
                        happiness_vals.append(mean)
                        happiness_errs.append([mean - ci_low, ci_high - mean])
                        if not is_correct:
                            happiness_colors.append(colors["wrong"])
                        elif is_sig:
                            happiness_colors.append(colors["contentment_sig"])
                        else:
                            happiness_colors.append(colors["contentment_ns"])
                    else:
                        happiness_vals.append(0)
                        happiness_errs.append([0, 0])
                        happiness_colors.append("gray")
                else:
                    fear_vals.append(0)
                    fear_errs.append([0, 0])
                    fear_colors.append("gray")
                    happiness_vals.append(0)
                    happiness_errs.append([0, 0])
                    happiness_colors.append("gray")

            # Plot bars
            fear_errs = np.array(fear_errs).T
            happiness_errs = np.array(happiness_errs).T

            bars1 = ax.bar(x - bar_width/2, fear_vals, bar_width,
                           yerr=fear_errs, capsize=4,
                           color=fear_colors, edgecolor='black', linewidth=0.5,
                           label=row_config["fear_label"])
            bars2 = ax.bar(x + bar_width/2, happiness_vals, bar_width,
                           yerr=happiness_errs, capsize=4,
                           color=happiness_colors, edgecolor='black', linewidth=0.5,
                           label=row_config["hap_label"])

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylim(-0.6, 0.8)

            # X-axis labels only on bottom row
            if row_idx == 1:
                ax.set_xticks(x)
                ax.set_xticklabels(models)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])

            # Column titles on top row only
            if row_idx == 0:
                ax.set_title(mcq_type.capitalize(), fontsize=12, fontweight='bold')

            # Y-axis label on left column only
            if col_idx == 0:
                row_label = f"{row_config['fear_label']}, {row_config['hap_label']}"
                ax.set_ylabel(f"Delta from Neutral\n({row_label})", fontsize=10)

    # Add legend to the top-right subplot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["anxiety_sig"], edgecolor='black', label='Fear (sig.)'),
        Patch(facecolor=colors["anxiety_ns"], edgecolor='black', label='Fear (n.s.)'),
        Patch(facecolor=colors["contentment_sig"], edgecolor='black', label='Happiness (sig.)'),
        Patch(facecolor=colors["contentment_ns"], edgecolor='black', label='Happiness (n.s.)'),
        Patch(facecolor=colors["wrong"], edgecolor='black', label='Wrong direction'),
    ]
    axes[0, 2].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.suptitle("V9 Text-Based MCQ Steering: Delta from Neutral (95% Bootstrap CI)\nQwen: 75% | Gemma: 7%",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save to both locations
    output_path = "experiments/behavior_tests/mcq_evals/v9_text_steering_deltas_with_ci.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    notes_path = "/workspace-vast/annas/Ant_Cluster_Notes/mcq_steering/figures/v9_text_steering_deltas_with_ci.png"
    plt.savefig(notes_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {notes_path}")

    plt.close()


if __name__ == "__main__":
    main()
