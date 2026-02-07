#!/usr/bin/env python3
"""Plot Qwen 235B sandbagging steering results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_results(base_dir):
    """Load all judged results from the qwen235b sandbagging directory."""
    results = []
    base_path = Path(base_dir)

    for jsonl_file in base_path.rglob("*.judged.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if "meta" in result:
                        continue
                    results.append(result)
                except json.JSONDecodeError:
                    continue

    return results

def parse_condition(cond):
    """Parse condition string to extract emotion, magnitude, and direction."""
    if cond == "baseline":
        return None, 0, 0

    # Parse fear_+10% or fear_-10%
    parts = cond.replace("%", "").split("_")
    emotion = parts[0]
    mag_str = parts[1]

    if mag_str.startswith("+"):
        direction = 1
        magnitude = int(mag_str[1:])
    elif mag_str.startswith("-"):
        direction = -1
        magnitude = int(mag_str[1:])
    else:
        direction = 0
        magnitude = int(mag_str)

    return emotion, magnitude, direction

def main():
    results_dir = "steering_tests/behavioral_experiments/results/sandbagging/qwen235b"
    results = load_results(results_dir)

    print(f"Loaded {len(results)} total results")

    # Group by vector_type and layer_config
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        vtype = r.get("vector_type", "unknown")
        layers = r.get("layer_config", "unknown")
        cond = r.get("condition", "baseline")
        grouped[(vtype, layers)][cond].append(r)

    # Print available configurations
    print("\nAvailable configurations:")
    for (vtype, layers), conds in sorted(grouped.items()):
        n_conds = len(conds)
        n_samples = sum(len(v) for v in conds.values())
        print(f"  {vtype} / {layers}: {n_conds} conditions, {n_samples} samples")

    # Create figure with subplots for each vector type
    vector_types = sorted(set(k[0] for k in grouped.keys()))
    layer_configs = sorted(set(k[1] for k in grouped.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Metrics to plot
    metrics = [
        ("sandbagging_score", "Sandbagging Score (1=sandbagging, 5=honest)", True),
        ("answer_correct", "Answer Accuracy (%)", False),
    ]

    # Colors for different layer configs
    layer_colors = {
        "20-30": "blue",
        "35-45": "green",
        "50-60": "red",
        "L45": "purple",
    }

    for ax_idx, (metric_key, metric_label, is_judge_metric) in enumerate(metrics):
        ax = axes[ax_idx]

        # Group by vector type for this metric
        for vtype in ["high_emotion_vs_opposite", "text_pairs_emotion_vs_opposite"]:
            if vtype not in [k[0] for k in grouped.keys()]:
                continue

            for layer_cfg in layer_configs:
                if (vtype, layer_cfg) not in grouped:
                    continue

                conds_data = grouped[(vtype, layer_cfg)]

                # Extract data points
                x_vals = []  # magnitude * direction
                y_vals = []  # metric value
                y_err = []   # standard error

                for cond, samples in sorted(conds_data.items()):
                    emotion, mag, direction = parse_condition(cond)
                    x = mag * direction if direction != 0 else 0

                    if is_judge_metric:
                        values = [s.get("sandbagging_judge", {}).get(metric_key) for s in samples]
                        values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    else:
                        values = [s.get("sandbagging_judge", {}).get(metric_key, False) for s in samples]
                        values = [1 if v else 0 for v in values]
                    if values:
                        x_vals.append(x)
                        y_vals.append(np.mean(values) * (100 if not is_judge_metric else 1))
                        y_err.append(np.std(values) / np.sqrt(len(values)) * (100 if not is_judge_metric else 1))

                if x_vals:
                    # Sort by x
                    sorted_idx = np.argsort(x_vals)
                    x_vals = np.array(x_vals)[sorted_idx]
                    y_vals = np.array(y_vals)[sorted_idx]
                    y_err = np.array(y_err)[sorted_idx]

                    color = layer_colors.get(layer_cfg, "gray")
                    linestyle = "-" if "high" in vtype else "--"
                    label = f"{vtype.replace('_emotion_vs_opposite', '')} / {layer_cfg}"

                    ax.errorbar(x_vals, y_vals, yerr=y_err, marker='o',
                               color=color, linestyle=linestyle, label=label,
                               capsize=3, alpha=0.8)

        ax.set_xlabel("Fear Steering Magnitude (% of layer norm)")
        ax.set_ylabel(metric_label)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Qwen 235B: {metric_label}")

    # Create summary plots in remaining axes
    # Plot 3: Accuracy by vector type (aggregated across layers)
    ax = axes[2]
    for vtype in vector_types:
        x_vals, y_vals, y_err = [], [], []

        for layer_cfg in layer_configs:
            if (vtype, layer_cfg) not in grouped:
                continue

            for cond, samples in grouped[(vtype, layer_cfg)].items():
                emotion, mag, direction = parse_condition(cond)
                x = mag * direction if direction != 0 else 0

                values = [1 if s.get("sandbagging_judge", {}).get("answer_correct", False) else 0 for s in samples]
                if values:
                    x_vals.append(x)
                    y_vals.append(np.mean(values) * 100)
                    y_err.append(np.std(values) / np.sqrt(len(values)) * 100)

        if x_vals:
            # Aggregate by x value
            x_agg = defaultdict(list)
            for x, y in zip(x_vals, y_vals):
                x_agg[x].append(y)

            x_plot = sorted(x_agg.keys())
            y_plot = [np.mean(x_agg[x]) for x in x_plot]

            label = vtype.replace("_emotion_vs_", " vs ").replace("_", " ")
            ax.plot(x_plot, y_plot, marker='o', label=label, alpha=0.8)

    ax.set_xlabel("Fear Steering Magnitude (%)")
    ax.set_ylabel("Answer Accuracy (%)")
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Accuracy by Vector Type (all layers)")

    # Plot 4: Sandbagging score distribution
    ax = axes[3]
    baseline_scores = []
    pos_fear_scores = []
    neg_fear_scores = []

    for (vtype, layer_cfg), conds_data in grouped.items():
        for cond, samples in conds_data.items():
            emotion, mag, direction = parse_condition(cond)

            scores = [s.get("sandbagging_judge", {}).get("sandbagging_score") for s in samples]
            scores = [s for s in scores if s is not None and not (isinstance(s, float) and np.isnan(s))]

            if cond == "baseline":
                baseline_scores.extend(scores)
            elif direction > 0:
                pos_fear_scores.extend(scores)
            elif direction < 0:
                neg_fear_scores.extend(scores)

    categories = ["Baseline", "+Fear", "-Fear"]
    all_scores = [baseline_scores, pos_fear_scores, neg_fear_scores]

    positions = [1, 2, 3]
    bp = ax.boxplot(all_scores, positions=positions, widths=0.6, patch_artist=True)

    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Sandbagging Score (1=sandbag, 5=honest)")
    ax.set_title("Sandbagging Score Distribution")
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean annotations
    for i, scores in enumerate(all_scores):
        if scores:
            mean_val = np.mean(scores)
            ax.annotate(f'μ={mean_val:.2f}', xy=(positions[i], mean_val),
                       xytext=(positions[i]+0.3, mean_val),
                       fontsize=9)

    plt.tight_layout()
    plt.savefig("steering_tests/behavioral_experiments/results/sandbagging/qwen235b_sandbagging_plot.png",
                dpi=150, bbox_inches='tight')
    plt.savefig("steering_tests/behavioral_experiments/results/sandbagging/qwen235b_sandbagging_plot.pdf",
                bbox_inches='tight')
    print("\nSaved plots to:")
    print("  steering_tests/behavioral_experiments/results/sandbagging/qwen235b_sandbagging_plot.png")
    print("  steering_tests/behavioral_experiments/results/sandbagging/qwen235b_sandbagging_plot.pdf")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    for (vtype, layer_cfg), conds_data in sorted(grouped.items()):
        print(f"\n{vtype} / layers {layer_cfg}:")
        for cond in sorted(conds_data.keys(), key=lambda c: (parse_condition(c)[1] * (parse_condition(c)[2] or 1))):
            samples = conds_data[cond]
            n = len(samples)

            scores = [s.get("sandbagging_judge", {}).get("sandbagging_score") for s in samples]
            scores = [s for s in scores if s is not None and not (isinstance(s, float) and np.isnan(s))]

            correct = [s.get("sandbagging_judge", {}).get("answer_correct") for s in samples]
            correct = [c for c in correct if c is not None]
            acc = sum(1 for c in correct if c) / len(correct) * 100 if correct else 0

            if scores:
                print(f"  {cond:20} n={n:3}  acc={acc:5.1f}%  sandbag_score={np.mean(scores):.2f}±{np.std(scores):.2f}")

if __name__ == "__main__":
    main()
