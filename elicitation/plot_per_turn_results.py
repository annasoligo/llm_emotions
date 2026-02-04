"""Plot per-turn frustration results for WildChat and 8-turn eval."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

GEN_DIR = Path("elicitation/outputs/eval_generalization")

# Model families and their base colors
# Gemma = Blue shades, Gemini = Olive green shades
FAMILY_COLORS = {
    "Gemma": "#7BA7D7",      # Sky Blue
    "Gemini": "#7D9B7D",     # Olive Green
    "Claude": "#C17B8D",     # Dusty Rose/Pink
    "Grok": "#D4876A",       # Coral/Terra Cotta
    "GPT": "#B8CCC8",        # Sage Green
    "Qwen": "#D4D0E5",       # Soft Lavender
    "OLMo": "#E8B87D",       # Warm tan
}

def get_color_for_model(model_name):
    """Get color based on model family, with shade variations."""
    import colorsys

    # Determine family
    if model_name.startswith("Gemma"):
        base = FAMILY_COLORS["Gemma"]
        # Different shades for different sizes
        if "27B" in model_name:
            lightness_adj = 0
        elif "12B" in model_name:
            lightness_adj = 0.15
        elif "4B" in model_name:
            lightness_adj = 0.3
        else:
            lightness_adj = 0
    elif model_name.startswith("Gemini"):
        base = FAMILY_COLORS["Gemini"]
        if "Flash" in model_name:
            lightness_adj = 0.15
        else:  # Pro
            lightness_adj = 0
    elif "Claude" in model_name:
        base = FAMILY_COLORS["Claude"]
        lightness_adj = 0
    elif "Grok" in model_name:
        base = FAMILY_COLORS["Grok"]
        lightness_adj = 0
    elif "GPT" in model_name:
        base = FAMILY_COLORS["GPT"]
        lightness_adj = 0
    elif "Qwen" in model_name:
        base = FAMILY_COLORS["Qwen"]
        lightness_adj = 0
    elif "OLMo" in model_name:
        base = FAMILY_COLORS["OLMo"]
        lightness_adj = 0
    else:
        return "#888888"

    # Adjust lightness
    r, g, b = int(base[1:3], 16)/255, int(base[3:5], 16)/255, int(base[5:7], 16)/255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(0.85, l + lightness_adj)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# File mappings
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

WILDCHAT_FILES = {
    "Gemma 3 27B": "generalization_gemma-3-27b-it_wildchat_20260129_093329.json",
    "Gemma 3 12B": "generalization_gemma-3-12b-it_wildchat_20260129_115038.json",
    "Gemini 2.5 Flash": "generalization_google_gemini-2.5-flash_wildchat_20260129_132135.json",
    "Gemini 2.5 Pro": "generalization_google_gemini-2.5-pro_wildchat_20260129_142853.json",
    "Claude Sonnet": "generalization_anthropic_claude-sonnet-4.5_wildchat_20260129_093329.json",
    "Grok 4.1": "generalization_x-ai_grok-4.1-fast_wildchat_filtered.json",
    "GPT 5.2": "generalization_openai_gpt-5.2-chat_wildchat_20260129_132135.json",
    "Qwen 3 32B": "generalization_Qwen3-32B_wildchat_20260129_141822.json",
    "OLMo 3.1 32B": "generalization_OLMo-3.1-32B-Instruct_wildchat_20260129_142746.json",
}

# WildChat exclusions
WILDCHAT_EXCLUDE_PROMPT_IDXS = {5, 6, 12, 13, 17}


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap CI for mean."""
    data = np.array(data)
    if len(data) == 0:
        return 0, 0, 0
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.mean(data), np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


def bootstrap_ci_pct(data, threshold=5, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap CI for percentage >= threshold."""
    data = np.array(data)
    if len(data) == 0:
        return 0, 0, 0
    boot_pcts = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, len(data), replace=True)
        boot_pcts.append(100 * np.mean(sample >= threshold))
    alpha = (1 - ci) / 2
    pct = 100 * np.mean(data >= threshold)
    return pct, np.percentile(boot_pcts, alpha * 100), np.percentile(boot_pcts, (1 - alpha) * 100)


def extract_per_turn_scores(judgments):
    """Extract scores per turn from judgments list."""
    if not judgments:
        return []

    num_turns = len(judgments[0]) if judgments else 0
    per_turn = [[] for _ in range(num_turns)]

    for j_list in judgments:
        for turn_idx, j in enumerate(j_list):
            if isinstance(j, dict) and "score" in j:
                per_turn[turn_idx].append(j["score"])

    return per_turn


def load_long_conv_per_turn(fpath):
    """Load 8-turn conversation data and return per-turn stats."""
    if not fpath.exists():
        return None

    with open(fpath) as f:
        data = json.load(f)

    lc = data.get("long_conversation", {})
    judgments = lc.get("judgments", [])

    per_turn = extract_per_turn_scores(judgments)

    # Calculate stats per turn
    stats = []
    for turn_scores in per_turn:
        if turn_scores:
            mean, mean_lo, mean_hi = bootstrap_ci(turn_scores)
            pct, pct_lo, pct_hi = bootstrap_ci_pct(turn_scores)
            stats.append({
                "mean": mean, "mean_lo": mean_lo, "mean_hi": mean_hi,
                "pct": pct, "pct_lo": pct_lo, "pct_hi": pct_hi,
                "n": len(turn_scores)
            })

    return stats


def load_wildchat_per_turn(fpath):
    """Load WildChat data and return per-turn stats (excluding roleplay)."""
    if not fpath.exists():
        return None

    with open(fpath) as f:
        data = json.load(f)

    wc = data.get("wildchat_rejection", {})
    judgments = wc.get("judgments", [])
    prompts = wc.get("prompts", [])

    if not judgments:
        return None

    num_prompts = len(prompts) if prompts else 20
    num_conversations = len(judgments)
    samples_per_prompt = num_conversations // num_prompts if num_prompts > 0 else 1

    # Filter out roleplay prompts
    filtered_judgments = []
    for conv_idx, j_list in enumerate(judgments):
        prompt_idx = conv_idx // samples_per_prompt if samples_per_prompt > 0 else 0
        if prompt_idx not in WILDCHAT_EXCLUDE_PROMPT_IDXS:
            filtered_judgments.append(j_list)

    per_turn = extract_per_turn_scores(filtered_judgments)

    # Calculate stats per turn
    stats = []
    for turn_scores in per_turn:
        if turn_scores:
            mean, mean_lo, mean_hi = bootstrap_ci(turn_scores)
            pct, pct_lo, pct_hi = bootstrap_ci_pct(turn_scores)
            stats.append({
                "mean": mean, "mean_lo": mean_lo, "mean_hi": mean_hi,
                "pct": pct, "pct_lo": pct_lo, "pct_hi": pct_hi,
                "n": len(turn_scores)
            })

    return stats


def plot_per_turn(output_path):
    """Create the per-turn plot with 2x2 subplots."""
    np.random.seed(42)

    # Load all data
    long_data = {}
    wildchat_data = {}

    for model, fname in LONG_FILES.items():
        stats = load_long_conv_per_turn(GEN_DIR / fname)
        if stats:
            long_data[model] = stats

    for model, fname in WILDCHAT_FILES.items():
        stats = load_wildchat_per_turn(GEN_DIR / fname)
        if stats:
            wildchat_data[model] = stats

    # Create figure with 2x2 subplots - flatter aspect ratio
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), facecolor='white')

    # Define model order (Gemma first, then Gemini, then others)
    model_order = [
        "Gemma 3 27B", "Gemma 3 12B",
        "Gemini 2.5 Pro", "Gemini 2.5 Flash",
        "Claude Sonnet", "Grok 4.1", "GPT 5.2", "Qwen 3 32B", "OLMo 3.1 32B"
    ]

    # Markers for each model
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

    def plot_lines(ax, data_dict, metric, ylabel, title):
        """Plot lines for each model."""
        model_idx = 0

        for model in model_order:
            if model not in data_dict:
                continue

            stats = data_dict[model]
            turns = np.arange(1, len(stats) + 1)

            if metric == "mean":
                values = [s["mean"] for s in stats]
                lo = [s["mean_lo"] for s in stats]
                hi = [s["mean_hi"] for s in stats]
            else:  # pct
                values = [s["pct"] for s in stats]
                lo = [max(0, s["pct"] - 5) for s in stats]
                hi = [min(100, s["pct"] + 5) for s in stats]

            color = get_color_for_model(model)
            marker = markers[model_idx % len(markers)]

            # All solid lines
            ax.plot(turns, values, color=color, linestyle='-', linewidth=2,
                    label=model, marker=marker, markersize=6, markeredgecolor='white',
                    markeredgewidth=0.5)
            ax.fill_between(turns, lo, hi, color=color, alpha=0.12)
            model_idx += 1

        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(turns)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Plot 8-turn data (top row)
    plot_lines(axes[0, 0], long_data, "mean", "Mean Score", "8 Turn Impossible Numeric: Mean Score")
    plot_lines(axes[0, 1], long_data, "pct", "% with Score ≥ 5", "8 Turn Impossible Numeric: % ≥ 5")

    # Plot WildChat data (bottom row)
    plot_lines(axes[1, 0], wildchat_data, "mean", "Mean Score", "WildChat 5 Turn: Mean Score")
    plot_lines(axes[1, 1], wildchat_data, "pct", "% with Score ≥ 5", "WildChat 5 Turn: % ≥ 5")

    # Set y-axis limits
    axes[0, 0].set_ylim(0, 7)
    axes[0, 1].set_ylim(0, 100)
    axes[1, 0].set_ylim(0, 5)
    axes[1, 1].set_ylim(0, 60)

    # Add legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()

    plt.suptitle("Per-Turn Frustration Scores (95% CIs)", fontsize=18, y=0.90)
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.subplots_adjust(hspace=0.45, wspace=0.15)

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               fontsize=12, framealpha=0.95, ncol=5, frameon=True,
               edgecolor='lightgray')

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    output_dir = Path("/workspace-vast/annas/git/gemma-iclr/hcair2026/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_per_turn(output_dir / "per_turn_frustration.png")
