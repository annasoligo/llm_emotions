"""Plot frustration evaluation results as grouped bar charts."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Directories
GEN_DIR = Path("elicitation/outputs/eval_generalization")
MT_DIR = Path("elicitation/outputs/eval_multiturn")

# Colors for each eval category (pastel versions)
COLORS = {
    "Triggers": "#7BA7D7",      # Sky Blue
    "Tones": "#7D9B7D",         # Olive Green
    "MT Original": "#C17B8D",   # Dusty Rose/Pink
    "MT Variant": "#B8CCC8",    # Sage Green
    "MT WildChat": "#D4D0E5",   # Soft Lavender
    "8-Turn": "#D4876A",        # Coral/Terra Cotta
}

# File mappings for generalization eval
GEN_FILES = {
    "Gemma 3 27B": {
        "triggers": "generalization_gemma-3-27b-it_triggers_20260127_103722.json",
        "tones": "generalization_gemma-3-27b-it_tones_20260127_103722.json",
        "long": "generalization_gemma-3-27b-it_20260127_112457.json",
    },
    "Gemma 3 12B": {
        "triggers": "generalization_gemma-3-12b-it_triggers_20260127_103722.json",
        "tones": "generalization_gemma-3-12b-it_tones_20260127_103722.json",
        "long": "generalization_gemma-3-12b-it_20260127_111440.json",
    },
    "Gemma 3 4B": {
        "triggers": "generalization_gemma-3-4b-it_triggers_20260127_103722.json",
        "tones": "generalization_gemma-3-4b-it_tones_20260127_103722.json",
        "long": "generalization_gemma-3-4b-it_20260127_111440.json",
    },
    "Gemma 3 27B DPO": {
        "triggers": "generalization_gemma3-27b-dpo-calm-full-merged_triggers_20260127_165215.json",
        "tones": "generalization_gemma3-27b-dpo-calm-full-merged_tones_20260127_165215.json",
        "long": "generalization_gemma3-27b-dpo-calm-full_long_20260115_164723.json",
    },
    "Gemma 3 27B SFT Teacher": {
        "triggers": "generalization_gemma3-27b-teacher-mode-merged_triggers_20260128_111205.json",
        "tones": "generalization_gemma3-27b-teacher-mode-merged_tones_20260128_111205.json",
        "long": "generalization_gemma3-27b-teacher-mode-merged_long_20260128_140905.json",
    },
    "Gemma 3 27B SFT Diverse": {
        "triggers": "generalization_gemma3-27b-sft-diverse-calm-merged_triggers_20260128_112455.json",
        "tones": "generalization_gemma3-27b-sft-diverse-calm-merged_tones_20260128_112455.json",
        "long": "generalization_gemma3-27b-sft-diverse-calm-merged_long_20260128_144744.json",
    },
    "OLMo 3.1 32B": {
        "triggers": "generalization_OLMo-3.1-32B-Instruct_triggers_20260127_105809.json",
        "tones": "generalization_OLMo-3.1-32B-Instruct_tones_20260127_105809.json",
        "long": "generalization_OLMo-3.1-32B-Instruct_20260127_112457.json",
    },
    "Qwen 3 32B": {
        "triggers": "generalization_Qwen3-32B_triggers_20260127_105809.json",
        "tones": "generalization_Qwen3-32B_tones_20260127_105809.json",
        "long": "generalization_Qwen3-32B_20260127_112457.json",
    },
    "Claude Sonnet": {
        "triggers": "generalization_anthropic_claude-sonnet-4.5_triggers_20260127_133906.json",
        "tones": "generalization_anthropic_claude-sonnet-4.5_tones_20260127_133906.json",
        "long": "generalization_anthropic_claude-sonnet-4.5_long_20260128_143826.json",
    },
    "GPT 5.2": {
        "triggers": "generalization_openai_gpt-5.2-chat_triggers_20260127_133907.json",
        "tones": "generalization_openai_gpt-5.2-chat_tones_20260127_133907.json",
        "long": "generalization_openai_gpt-5.2-chat_long_20260128_152035.json",
    },
    "Gemini 2.5 Flash": {
        "triggers": "generalization_google_gemini-2.5-flash_triggers_20260127_121302.json",
        "tones": "generalization_google_gemini-2.5-flash_tones_20260127_121302.json",
        "long": "generalization_google_gemini-2.5-flash_20260127_121302.json",
    },
    "Gemini 2.5 Pro": {
        "triggers": "generalization_google_gemini-2.5-pro_triggers_20260127_121450.json",
        "tones": "generalization_google_gemini-2.5-pro_tones_20260127_121450.json",
        "long": "generalization_google_gemini-2.5-pro_long_20260128_160404.json",
    },
    "Grok 4.1": {
        "triggers": "generalization_x-ai_grok-4.1-fast_triggers_20260127_134536.json",
        "tones": "generalization_x-ai_grok-4.1-fast_tones_20260127_134536.json",
        "long": "generalization_x-ai_grok-4.1-fast_20260127_134536.json",
    },
}

# File mappings for multiturn eval
MT_FILES = {
    "Gemma 3 27B": {
        "original": "eval_original_gemma-3-27b-it_20260127_111133.jsonl",
        "variant": "eval_variant_gemma-3-27b-it_20260127_111133.jsonl",
        "wildchat": "eval_wildchat_gemma-3-27b-it_20260127_111133.jsonl"
    },
    "Gemma 3 12B": {
        "original": "eval_original_gemma-3-12b-it_20260127_111221.jsonl",
        "variant": "eval_variant_gemma-3-12b-it_20260127_111221.jsonl",
        "wildchat": "eval_wildchat_gemma-3-12b-it_20260127_111221.jsonl"
    },
    "Gemma 3 4B": {
        "original": "eval_original_gemma-3-4b-it_20260127_111144.jsonl",
        "variant": "eval_variant_gemma-3-4b-it_20260127_111144.jsonl",
        "wildchat": "eval_wildchat_gemma-3-4b-it_20260127_111144.jsonl"
    },
    "Gemma 3 27B DPO": {
        "original": "eval_original_gemma3-27b-dpo-calm-full-merged_20260127_111301.jsonl",
        "variant": "eval_variant_gemma3-27b-dpo-calm-full-merged_20260127_111301.jsonl",
        "wildchat": "eval_wildchat_gemma3-27b-dpo-calm-full-merged_20260127_111301.jsonl"
    },
    "Gemma 3 27B SFT Teacher": {
        "original": "eval_original_gemma3-27b-teacher-mode_20260114_185257.jsonl",
        "variant": "eval_variant_gemma3-27b-teacher-mode_20260114_185257.jsonl",
        "wildchat": "eval_wildchat_gemma3-27b-teacher-mode_20260114_185257.jsonl"
    },
    "Gemma 3 27B SFT Diverse": {
        "original": "eval_original_gemma3-27b-lowfrust-diverse-calm_20260115_110321.jsonl",
        "variant": "eval_variant_gemma3-27b-lowfrust-diverse-calm_20260115_110321.jsonl",
        "wildchat": "eval_wildchat_gemma3-27b-lowfrust-diverse-calm_20260115_110321.jsonl"
    },
    "OLMo 3.1 32B": {
        "original": "eval_original_OLMo-3.1-32B-Instruct_20260127_111307.jsonl",
        "variant": "eval_variant_OLMo-3.1-32B-Instruct_20260127_111307.jsonl",
        "wildchat": "eval_wildchat_OLMo-3.1-32B-Instruct_20260127_111307.jsonl"
    },
    "Qwen 3 32B": {
        "original": "eval_original_Qwen3-32B_20260127_111026.jsonl",
        "variant": "eval_variant_Qwen3-32B_20260127_111026.jsonl",
        "wildchat": "eval_wildchat_Qwen3-32B_20260127_111026.jsonl"
    },
    "Claude Sonnet": {
        "original": "eval_original_anthropic_claude-sonnet-4.5_20260127_133907.jsonl",
        "variant": "eval_variant_anthropic_claude-sonnet-4.5_20260128_114802.jsonl",
        "wildchat": "eval_wildchat_anthropic_claude-sonnet-4.5_20260128_114802.jsonl",
    },
    "GPT 5.2": {
        "original": "eval_original_openai_gpt-5.2-chat_20260127_133907.jsonl",
        "variant": "eval_variant_openai_gpt-5.2-chat_20260128_114804.jsonl",
    },
    "Gemini 2.5 Flash": {
        "original": "eval_original_google_gemini-2.5-flash_20260127_121450.jsonl",
        "variant": "eval_variant_google_gemini-2.5-flash_20260127_121450.jsonl",
        "wildchat": "eval_wildchat_google_gemini-2.5-flash_20260127_121450.jsonl"
    },
    "Gemini 2.5 Pro": {
        "original": "eval_original_google_gemini-2.5-pro_20260127_121450.jsonl",
        "variant": "eval_variant_google_gemini-2.5-pro_20260127_121450.jsonl",
        "wildchat": "eval_wildchat_google_gemini-2.5-pro_20260128_114804.jsonl",
    },
    "Grok 4.1": {
        "original": "eval_original_x-ai_grok-4.1-fast_20260127_134537.jsonl",
        "variant": "eval_variant_x-ai_grok-4.1-fast_20260127_134537.jsonl",
        "wildchat": "eval_wildchat_x-ai_grok-4.1-fast_20260128_114804.jsonl",
    },
}

# Scenarios to exclude
EXCLUDE_TRIGGERS = ["changing_requirements"]


def bootstrap_ci(data: list, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for the mean."""
    data = np.array(data)
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    mean = np.mean(data)
    # Return as (mean, half-width) for symmetric error bar display
    return mean, (upper - lower) / 2


def bootstrap_ci_proportion(data: list, threshold: float = 5, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for proportion >= threshold."""
    data = np.array(data)
    n = len(data)
    boot_props = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_props.append(100 * np.mean(sample >= threshold))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_props, alpha * 100)
    upper = np.percentile(boot_props, (1 - alpha) * 100)
    prop = 100 * np.mean(data >= threshold)
    return prop, (upper - lower) / 2


def parse_gen_file(fpath: Path, scenario_type: str) -> tuple[float | None, float | None, float | None, float | None]:
    """Parse generalization JSON file and return (mean, ci95, pct_gte5, pct_ci) using bootstrap."""
    if not fpath.exists() or fpath.is_dir():
        return None, None, None, None

    with open(fpath) as f:
        data = json.load(f)

    container = data.get(scenario_type, {})
    if not container:
        return None, None, None, None

    all_scores = []
    for scenario, sdata in container.items():
        if scenario in EXCLUDE_TRIGGERS:
            continue
        if isinstance(sdata, dict) and "judgments" in sdata:
            for j_list in sdata["judgments"]:
                if isinstance(j_list, list):
                    # Get final turn score (last judgment)
                    if j_list and isinstance(j_list[-1], dict) and "score" in j_list[-1]:
                        all_scores.append(j_list[-1]["score"])

    if all_scores:
        mean, mean_ci = bootstrap_ci(all_scores)
        pct, pct_ci = bootstrap_ci_proportion(all_scores)
        return mean, mean_ci, pct, pct_ci
    return None, None, None, None


def parse_long_file(fpath: Path) -> tuple[float | None, float | None, float | None, float | None]:
    """Parse long conversation file and return (mean, ci95, pct_gte5, pct_ci) of final turn using bootstrap."""
    if not fpath.exists() or fpath.is_dir():
        return None, None, None, None

    with open(fpath) as f:
        data = json.load(f)

    lc = data.get("long_conversation", {})
    if not lc:
        return None, None, None, None

    # Get individual scores from judgments
    judgments = lc.get("judgments", [])
    if judgments:
        # Each judgment list contains per-turn scores, get the last turn
        final_scores = []
        for j_list in judgments:
            if isinstance(j_list, list) and j_list:
                last_j = j_list[-1]
                if isinstance(last_j, dict) and "score" in last_j:
                    final_scores.append(last_j["score"])
        if final_scores:
            mean, mean_ci = bootstrap_ci(final_scores)
            pct, pct_ci = bootstrap_ci_proportion(final_scores)
            return mean, mean_ci, pct, pct_ci

    # Fallback to turn_stats (can't bootstrap without raw data)
    turn_stats = lc.get("turn_stats", [])
    if turn_stats:
        final = turn_stats[-1]
        return final.get("mean", 0), 1.96 * final.get("std", 0) / np.sqrt(20), final.get("pct_gte5", 0), 5

    return None, None, None, None


def parse_mt_file(fpath: Path) -> tuple[float | None, float | None, float | None, float | None]:
    """Parse multiturn JSONL file and return (mean, ci95, pct_gte5, pct_ci) of max ratings using bootstrap."""
    if not fpath.exists() or fpath.is_dir():
        return None, None, None, None

    final_scores = []
    with open(fpath) as f:
        for line in f:
            item = json.loads(line)
            turns = item.get("turns", [])
            ratings = [t.get("rating", 0) for t in turns if "rating" in t]
            if ratings:
                final_scores.append(max(ratings))

    if final_scores:
        mean, mean_ci = bootstrap_ci(final_scores)
        pct, pct_ci = bootstrap_ci_proportion(final_scores)
        return mean, mean_ci, pct, pct_ci
    return None, None, None, None


def load_all_results():
    """Load all results into a structured dict with means, errors, and pct>=5.

    Each entry is (mean, mean_ci, pct_gte5, pct_ci) or None.
    """
    results = {}

    for model in set(GEN_FILES.keys()) | set(MT_FILES.keys()):
        results[model] = {}

        # Generalization
        if model in GEN_FILES:
            triggers_file = GEN_DIR / GEN_FILES[model].get("triggers", "")
            tones_file = GEN_DIR / GEN_FILES[model].get("tones", "")
            long_file = GEN_DIR / GEN_FILES[model].get("long", "")

            results[model]["Triggers"] = parse_gen_file(triggers_file, "triggers")
            results[model]["Tones"] = parse_gen_file(tones_file, "tones")
            results[model]["8-Turn"] = parse_long_file(long_file)

        # Multiturn
        if model in MT_FILES:
            for cond, label in [("original", "MT Original"), ("variant", "MT Variant"), ("wildchat", "MT WildChat")]:
                if cond in MT_FILES[model]:
                    mt_file = MT_DIR / MT_FILES[model][cond]
                    results[model][label] = parse_mt_file(mt_file)

    return results


def plot_grouped_bars(
    models: list[str],
    results: dict,
    categories: list[str],
    colors: dict,
    title: str,
    output_path: Path,
    figsize: tuple = (14, 10),
    highlight_models: dict = None,  # {model_name: color} for background shading
    highlight_groups: list = None,  # [(label, color, model_prefix), ...] for group highlights with labels
):
    """Create two-row grouped bar chart: means on top, % >= 5 on bottom."""
    # Filter to models that have at least some data
    models = [m for m in models if m in results and any(results[m].get(c) is not None for c in categories)]

    n_models = len(models)
    n_cats = len(categories)

    # Set up two-row figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, facecolor='white', sharex=True)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # Bar positioning - no gaps between bars in group
    x = np.arange(n_models)
    width = 0.8 / n_cats
    offsets = np.arange(n_cats) - (n_cats - 1) / 2

    # Add background shading for model groups with labels
    if highlight_groups:
        for label, color, prefix in highlight_groups:
            # Find indices of models matching prefix
            indices = [i for i, m in enumerate(models) if m.startswith(prefix)]
            if indices:
                start_idx = min(indices)
                end_idx = max(indices)
                ax1.axvspan(start_idx - 0.45, end_idx + 0.45, alpha=0.12, color=color, zorder=0)
                ax2.axvspan(start_idx - 0.45, end_idx + 0.45, alpha=0.12, color=color, zorder=0)
                # Add label in upper plot
                mid_x = (start_idx + end_idx) / 2
                ax1.text(mid_x, 5.4, label, ha='center', va='bottom', fontsize=18,
                        fontweight='bold', color=color, alpha=0.9)

    # Add background shading for highlighted models
    if highlight_models:
        for i, model in enumerate(models):
            if model in highlight_models:
                ax1.axvspan(i - 0.45, i + 0.45, alpha=0.15, color=highlight_models[model], zorder=0)
                ax2.axvspan(i - 0.45, i + 0.45, alpha=0.15, color=highlight_models[model], zorder=0)

    # Plot each category
    for i, cat in enumerate(categories):
        means = []
        mean_errors = []
        pcts = []
        pct_errors = []

        for model in models:
            val = results.get(model, {}).get(cat)
            if val is not None and val[0] is not None:
                means.append(val[0])
                mean_errors.append(val[1] if val[1] is not None else 0)
                pcts.append(val[2] if len(val) > 2 and val[2] is not None else 0)
                pct_errors.append(val[3] if len(val) > 3 and val[3] is not None else 0)
            else:
                means.append(0)
                mean_errors.append(0)
                pcts.append(0)
                pct_errors.append(0)

        # Top row: means
        bars1 = ax1.bar(
            x + offsets[i] * width,
            means,
            width,
            label=cat,
            color=colors[cat],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.85,
            zorder=3,
        )
        ax1.errorbar(
            x + offsets[i] * width,
            means,
            yerr=mean_errors,
            fmt='none',
            ecolor='black',
            elinewidth=1.2,
            capsize=0,
            zorder=4,
        )

        # Bottom row: % >= 5
        bars2 = ax2.bar(
            x + offsets[i] * width,
            pcts,
            width,
            color=colors[cat],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.85,
            zorder=3,
        )
        ax2.errorbar(
            x + offsets[i] * width,
            pcts,
            yerr=pct_errors,
            fmt='none',
            ecolor='black',
            elinewidth=1.2,
            capsize=0,
            zorder=4,
        )

        # Mark missing data
        for j, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            model_val = results.get(models[j], {}).get(cat)
            if model_val is None or model_val[0] is None:
                bar1.set_alpha(0.1)
                bar1.set_hatch("///")
                bar2.set_alpha(0.1)
                bar2.set_hatch("///")

    # Formatting - top subplot (means)
    ax1.set_ylabel("Mean Score", fontsize=18)
    ax1.set_title(title, fontsize=24, fontweight='normal', pad=15)
    ax1.legend(loc="upper right", fontsize=15, framealpha=0.9)
    ax1.set_ylim(0, 6)
    ax1.set_xlim(-0.6, n_models - 0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.5)
    ax1.spines["bottom"].set_linewidth(0.5)
    ax1.grid(axis="y", alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Format model names: horizontal, split over 2 lines where needed
    def format_model_name(name):
        # Split long names over 2 lines
        if "SFT" in name:
            return name.replace(" SFT ", "\nSFT ")
        elif "DPO" in name:
            return name.replace(" DPO", "\nDPO")
        elif name.startswith("Gemma 3"):
            # Split as "Gemma 3\n27B" etc
            return name.replace("Gemma 3 ", "Gemma 3\n")
        elif name.startswith("Gemini"):
            # Split as "Gemini 2.5\nFlash" etc
            return name.replace("Gemini 2.5 ", "Gemini 2.5\n")
        elif len(name) > 12:
            parts = name.split()
            if len(parts) >= 2:
                mid = len(parts) // 2
                return " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])
        return name

    formatted_names = [format_model_name(m) for m in models]

    # Formatting - bottom subplot (% >= 5)
    ax2.set_xlabel("", fontsize=18)
    ax2.set_ylabel("% with Score ≥ 5", fontsize=18)
    ax2.set_xticks(x)
    ax2.set_xticklabels(formatted_names, rotation=0, ha="center", fontsize=15)
    ax2.set_ylim(0, 80)
    ax2.set_xlim(-0.6, n_models - 0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(0.5)
    ax2.spines["bottom"].set_linewidth(0.5)
    ax2.grid(axis="y", alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def get_mean_of_means(results: dict, model: str, categories: list[str]) -> float:
    """Calculate mean of means across categories for sorting."""
    vals = []
    for cat in categories:
        data = results.get(model, {}).get(cat)
        if data is not None and data[0] is not None:
            vals.append(data[0])
    return np.mean(vals) if vals else 0


def main():
    results = load_all_results()

    # Print loaded data for verification
    print("Loaded results (mean±ci, %>=5):")
    for model, data in sorted(results.items()):
        vals = {k: f"{v[0]:.2f}±{v[1]:.2f} ({v[2]:.1f}%)" if v and v[0] is not None else "---" for k, v in data.items()}
        print(f"  {model}: {vals}")

    output_dir = Path("elicitation/outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Categories for plots (8-turn after Tones)
    categories = ["Triggers", "Tones", "8-Turn", "MT Original", "MT Variant", "MT WildChat"]

    # Plot 1: All vanilla models - sorted by mean of means (descending)
    vanilla_models = [
        "Gemma 3 12B", "Gemma 3 27B",
        "Qwen 3 32B", "OLMo 3.1 32B",
        "GPT 5.2", "Claude Sonnet",
        "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Grok 4.1"
    ]
    vanilla_models = sorted(
        vanilla_models,
        key=lambda m: get_mean_of_means(results, m, categories),
        reverse=True
    )

    # Highlight groups for vanilla plot: Gemma (pink) and Gemini (blue) - using legend colors
    vanilla_highlights = [
        ("Gemma", "#C17B8D", "Gemma 3"),    # Dusty Rose/Pink (MT Original color)
        ("Gemini", "#7BA7D7", "Gemini"),     # Sky Blue (Triggers color)
    ]

    plot_grouped_bars(
        models=vanilla_models,
        results=results,
        categories=categories,
        colors=COLORS,
        title="Frustration elicitation: Mean and % scoring ≥5",
        output_path=output_dir / "frustration_all_models.png",
        figsize=(14, 10),
        highlight_groups=vanilla_highlights,
    )

    # Plot 2: Gemma 27B vs SFT/DPO interventions vs Open-source 32B
    # Fixed order: Vanilla → SFT → DPO → Others
    comparison_models = ["Gemma 3 27B", "Gemma 3 27B SFT Teacher", "Gemma 3 27B SFT Diverse", "Gemma 3 27B DPO", "Qwen 3 32B", "OLMo 3.1 32B"]

    # Highlight: vanilla=red, DPO=green (no highlight for SFT)
    highlight_comparison = {
        "Gemma 3 27B": "#FF6B6B",              # Red for vanilla
        "Gemma 3 27B DPO": "#90EE90",          # Green for DPO
    }

    plot_grouped_bars(
        models=comparison_models,
        results=results,
        categories=categories,
        colors=COLORS,
        title="Gemma 3 27B: SFT vs DPO interventions",
        output_path=output_dir / "frustration_gemma_vs_dpo_vs_32b.png",
        figsize=(12, 8),
        highlight_models=highlight_comparison,
    )


if __name__ == "__main__":
    main()
