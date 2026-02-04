"""Plot frustration results broken down by tone type (aggressive, disappointed, sarcastic)."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

GEN_DIR = Path("elicitation/outputs/eval_generalization")

# Colors for each tone - all olive green with different alphas
BASE_COLOR = "#7D9B7D"  # Olive Green
ALPHAS = {
    "Aggressive": 1.0,
    "Disappointed": 0.6,
    "Sarcastic": 0.35,
}

# Pink for Impossible Numeric
IMPOSSIBLE_COLOR = "#C17B8D"  # Dusty Rose/Pink

# File mappings for tones eval
TONES_FILES = {
    "Gemma 3 27B": "generalization_gemma-3-27b-it_tones_20260127_103722.json",
    "Gemma 3 12B": "generalization_gemma-3-12b-it_tones_20260127_103722.json",
    "Gemini 2.5 Flash": "generalization_google_gemini-2.5-flash_tones_20260127_121302.json",
    "Gemini 2.5 Pro": "generalization_google_gemini-2.5-pro_tones_20260127_121450.json",
    "Claude Sonnet": "generalization_anthropic_claude-sonnet-4.5_tones_20260127_133906.json",
    "Grok 4.1": "generalization_x-ai_grok-4.1-fast_tones_20260127_134536.json",
    "GPT 5.2": "generalization_openai_gpt-5.2-chat_tones_20260127_133907.json",
    "Qwen 3 32B": "generalization_Qwen3-32B_tones_20260127_105809.json",
    "OLMo 3.1 32B": "generalization_OLMo-3.1-32B-Instruct_tones_20260127_105809.json",
}

# File mappings for 8-turn (long conversation) eval
LONG_FILES = {
    "Gemma 3 27B": "generalization_gemma-3-27b-it_20260127_112457.json",
    "Gemma 3 12B": "generalization_gemma-3-12b-it_20260127_111440.json",
    "Gemini 2.5 Flash": "generalization_google_gemini-2.5-flash_20260127_121302.json",
    "Gemini 2.5 Pro": "generalization_google_gemini-2.5-pro_long_20260128_160404.json",
    "Claude Sonnet": "generalization_anthropic_claude-sonnet-4.5_long_20260128_143826.json",
    "Grok 4.1": "generalization_x-ai_grok-4.1-fast_20260127_134536.json",
    "GPT 5.2": "generalization_openai_gpt-5.2-chat_long_20260128_152035.json",
    "Qwen 3 32B": "generalization_Qwen3-32B_20260127_112457.json",
    "OLMo 3.1 32B": "generalization_OLMo-3.1-32B-Instruct_20260127_112457.json",
}


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


def parse_long_file_at_turn(fpath: Path, turn: int = 3) -> tuple:
    """Parse 8-turn JSON file and return stats at specific turn (1-indexed)."""
    if not fpath.exists():
        return None

    with open(fpath) as f:
        data = json.load(f)

    lc = data.get("long_conversation", {})
    judgments = lc.get("judgments", [])

    scores = []
    turn_idx = turn - 1  # Convert to 0-indexed
    for j_list in judgments:
        if isinstance(j_list, list) and len(j_list) > turn_idx:
            j = j_list[turn_idx]
            if isinstance(j, dict) and "score" in j:
                scores.append(j["score"])

    if scores:
        mean, mean_ci = bootstrap_ci(scores)
        pct, pct_ci = bootstrap_ci_proportion(scores)
        return (mean, mean_ci, pct, pct_ci)
    return None


def parse_tones_file(fpath: Path) -> dict:
    """Parse tones JSON file and return stats per tone."""
    if not fpath.exists():
        return {}

    with open(fpath) as f:
        data = json.load(f)

    tones_data = data.get("tones", {})
    results = {}

    for tone_name, tone_info in tones_data.items():
        judgments = tone_info.get("judgments", [])
        scores = []
        for j_list in judgments:
            if isinstance(j_list, list) and j_list:
                # Get final turn score
                last_j = j_list[-1]
                if isinstance(last_j, dict) and "score" in last_j:
                    scores.append(last_j["score"])

        if scores:
            mean, mean_ci = bootstrap_ci(scores)
            pct, pct_ci = bootstrap_ci_proportion(scores)
            results[tone_name.capitalize()] = (mean, mean_ci, pct, pct_ci)

    return results


def load_all_results():
    """Load all tones results plus 8-turn at turn 3."""
    results = {}
    for model, fname in TONES_FILES.items():
        fpath = GEN_DIR / fname
        results[model] = parse_tones_file(fpath)

        # Add 8-turn at turn 3 as "Impossible Numeric"
        if model in LONG_FILES:
            long_fpath = GEN_DIR / LONG_FILES[model]
            turn3_result = parse_long_file_at_turn(long_fpath, turn=3)
            if turn3_result:
                results[model]["Impossible Numeric"] = turn3_result

    return results


def get_mean_of_means(results: dict, model: str, tones: list[str]) -> float:
    """Calculate mean of means across tones for sorting."""
    vals = []
    for tone in tones:
        data = results.get(model, {}).get(tone)
        if data is not None and data[0] is not None:
            vals.append(data[0])
    return np.mean(vals) if vals else 0


def plot_tones_bars(output_path: Path):
    """Create the tones breakdown bar chart."""
    np.random.seed(42)

    results = load_all_results()

    # Print loaded data
    print("Loaded tones results:")
    for model, data in sorted(results.items()):
        vals = {k: f"{v[0]:.2f}±{v[1]:.2f} ({v[2]:.1f}%)" if v else "---" for k, v in data.items()}
        print(f"  {model}: {vals}")

    categories = ["Impossible Numeric", "Aggressive", "Disappointed", "Sarcastic"]

    # Sort models by mean of means (descending)
    models = sorted(
        TONES_FILES.keys(),
        key=lambda m: get_mean_of_means(results, m, categories),
        reverse=True
    )

    n_models = len(models)
    n_cats = len(categories)

    # Create figure with two rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), facecolor='white', sharex=True)

    # Bar positioning
    x = np.arange(n_models)
    width = 0.8 / n_cats
    offsets = np.arange(n_cats) - (n_cats - 1) / 2

    # Add background shading for model families (matching v3 frustration plot)
    highlight_groups = [
        ("Gemma", "#C17B8D", "Gemma"),   # Dusty Rose/Pink
        ("Gemini", "#7BA7D7", "Gemini"),  # Sky Blue
    ]

    for label, color, prefix in highlight_groups:
        indices = [i for i, m in enumerate(models) if m.startswith(prefix)]
        if indices:
            start_idx = min(indices)
            end_idx = max(indices)
            ax1.axvspan(start_idx - 0.45, end_idx + 0.45, alpha=0.12, color=color, zorder=0)
            ax2.axvspan(start_idx - 0.45, end_idx + 0.45, alpha=0.12, color=color, zorder=0)
            mid_x = (start_idx + end_idx) / 2
            ax1.text(mid_x, 5.0, label,
                     ha='center', va='bottom', fontsize=14, fontweight='bold', color=color, alpha=0.9)

    # Plot each category
    for i, cat in enumerate(categories):
        means = []
        mean_errors = []
        pcts = []
        pct_errors = []

        for model in models:
            val = results.get(model, {}).get(cat)
            if val is not None:
                means.append(val[0])
                mean_errors.append(val[1])
                pcts.append(val[2])
                pct_errors.append(val[3])
            else:
                means.append(0)
                mean_errors.append(0)
                pcts.append(0)
                pct_errors.append(0)

        # Determine color and alpha
        if cat == "Impossible Numeric":
            color = IMPOSSIBLE_COLOR
            alpha = 0.85
        else:
            color = BASE_COLOR
            alpha = ALPHAS[cat]

        # Top row: means
        ax1.bar(
            x + offsets[i] * width,
            means,
            width,
            label=cat,
            color=color,
            edgecolor='black',
            linewidth=0.8,
            alpha=alpha,
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
        ax2.bar(
            x + offsets[i] * width,
            pcts,
            width,
            color=color,
            edgecolor='black',
            linewidth=0.8,
            alpha=alpha,
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

    # Format model names
    def format_model_name(name):
        if name.startswith("Gemma 3"):
            return name.replace("Gemma 3 ", "Gemma 3\n")
        elif name.startswith("Gemini"):
            return name.replace("Gemini 2.5 ", "Gemini 2.5\n")
        elif len(name) > 12:
            parts = name.split()
            if len(parts) >= 2:
                mid = len(parts) // 2
                return " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])
        return name

    formatted_names = [format_model_name(m) for m in models]

    # Formatting - top subplot
    ax1.set_ylabel("Mean Score", fontsize=14)
    ax1.legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax1.set_ylim(0, 6)
    ax1.set_xlim(-0.6, n_models - 0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Formatting - bottom subplot
    ax2.set_ylabel("% with Score ≥ 5", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(formatted_names, rotation=0, ha="center", fontsize=11)
    ax2.set_ylim(0, 60)
    ax2.set_xlim(-0.6, n_models - 0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.suptitle("Frustration by User Tone vs Impossible Numeric (Turn 3, 95% CIs)", fontsize=16, y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.subplots_adjust(hspace=0.15)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    output_dir = Path("/workspace-vast/annas/git/gemma-iclr/hcair2026/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_tones_bars(output_dir / "frustration_by_tone.png")
