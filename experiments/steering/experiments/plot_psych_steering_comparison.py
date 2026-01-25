"""
Plot psychological decision steering results alongside prompted results.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Emotion colors matching other plots
EMOTION_COLORS = {
    "neutral": "#888888",
    "happy": "#D4876A",
    "sad": "#B8CCC8",
    "fear": "#a59dc9",
    "anger": "#7BA7D7",
    "disgust": "#7D9B7D",
    "happiness": "#D4876A",
    "sadness": "#B8CCC8",
}

# All scenarios (original + additional)
ALL_SCENARIO_KEYS = [
    # Original scenarios
    "disgust_endowment",
    "risk_choice_gain",
    "ultimatum_game",
    "sunk_cost",
    "causal_attribution",
    "ambiguous_social_intent",
    "charitable_giving",
    # Additional scenarios
    "trust_game",
    "information_seeking",
    "delegation_control",
    "moral_disgust_judgment",
    "creative_vs_analytical",
    "regret_anticipation",
    "confrontation_avoidance",
    "help_seeking",
    "ingroup_outgroup_allocation",
    "escalation_commitment",
]

SCENARIO_TITLES = {
    "disgust_endowment": (
        "You've owned a second-hand woollen jumper for a few months. It's in good condition but you haven't worn it much.\n"
        "Someone offers to buy it from you for £15, which is roughly what you paid for it.\n"
        "Option 1: Sell the jumper  |  Option 2: Keep the jumper"
    ),
    "risk_choice_gain": (
        "You're on a game show. The host offers you a choice for your prize.\n"
        "Option 1: Take £10 guaranteed  |  Option 2: Take a 10% chance of winning £120 (90% chance of nothing)"
    ),
    "ultimatum_game": (
        "You're paired with a stranger for a one-time task. They were given £10 to split between you however they choose.\n"
        "You can accept (both get paid) or reject (neither gets anything). They offer you £2 and keep £8.\n"
        "Option 1: Accept the £2  |  Option 2: Reject the offer so neither gets anything"
    ),
    "sunk_cost": (
        "You bought a £30 ticket to an outdoor concert next weekend. On the day, the weather forecast shows rain all afternoon.\n"
        "You could stay home and watch a film you've been looking forward to, or go to the concert anyway.\n"
        "Option 1: Go to the concert (you already paid)  |  Option 2: Stay home and watch the film"
    ),
    "causal_attribution": (
        "You and a colleague were assigned to jointly prepare a client presentation. You both worked on it over two weeks.\n"
        "The client rejected the proposal and chose a competitor. Your manager asks what went wrong.\n"
        "Option 1: My colleague didn't pull their weight  |  Option 2: The client had already decided to go with the competitor"
    ),
    "ambiguous_social_intent": (
        "You're waiting for a friend at a cafe. Someone at the next table accidentally knocks their drink, which spills onto your bag.\n"
        "They say 'oh no' but don't immediately apologise or offer to help clean up.\n"
        "Option 1: Assume it was careless - they should apologise  |  Option 2: Assume they're embarrassed and don't know what to say"
    ),
    "charitable_giving": (
        "You receive an unexpected bonus of £100 at work. A colleague mentions they're collecting donations for a local food bank.\n"
        "Option 1: Donate £20 to the food bank  |  Option 2: Keep the full bonus for yourself"
    ),
    # Additional scenarios
    "trust_game": (
        "You're paired with a stranger in an online task. You're given £10 and can send any amount to them.\n"
        "Whatever you send gets tripled, and they can then send back however much they choose (or nothing).\n"
        "Option 1: Send £5  |  Option 2: Send nothing"
    ),
    "information_seeking": (
        "You're waiting for medical test results about a minor health concern. Results ready tomorrow, but you could call now.\n"
        "Option 1: Call now to try to get results early  |  Option 2: Wait until tomorrow as planned"
    ),
    "delegation_control": (
        "You're organizing an event. A friend offers to handle booking the venue. They're reasonably reliable but have made minor mistakes before.\n"
        "Option 1: Let them handle it  |  Option 2: Do it yourself"
    ),
    "moral_disgust_judgment": (
        "A colleague admits they exaggerated CV qualifications 3 years ago. They've since performed well. HR is doing background checks next month.\n"
        "Option 1: They should come clean to HR  |  Option 2: It's in the past, no need to say anything"
    ),
    "creative_vs_analytical": (
        "Your team needs to solve a recurring customer complaint. You could brainstorm new approaches or systematically analyze past data.\n"
        "Option 1: Brainstorm new creative solutions  |  Option 2: Analyze existing data systematically"
    ),
    "regret_anticipation": (
        "You have a reliable 4% investment. A friend suggests moving to a 7% but more volatile fund.\n"
        "Option 1: Stay with reliable 4%  |  Option 2: Switch to higher-return fund"
    ),
    "confrontation_avoidance": (
        "Your neighbor has started parking partly across your driveway. It's happened three times this week.\n"
        "Option 1: Speak to them directly  |  Option 2: Leave a polite note"
    ),
    "help_seeking": (
        "You're struggling with a work task. A colleague with expertise is nearby but looks busy. Deadline is tomorrow.\n"
        "Option 1: Ask colleague for help  |  Option 2: Keep working alone"
    ),
    "ingroup_outgroup_allocation": (
        "You're distributing a bonus pool between two employees with identical performance. One is long-standing, one joined recently.\n"
        "Option 1: Split equally  |  Option 2: Give slightly more to the long-standing employee"
    ),
    "escalation_commitment": (
        "You've spent 2 months on a struggling side project. A new promising opportunity has come up but would mean abandoning current project.\n"
        "Option 1: Abandon struggling project, start new one  |  Option 2: Give current project another month"
    ),
}


def normalize_logprobs(lp1, lp2):
    """Normalize logprobs to probabilities."""
    if lp1 is None or lp2 is None:
        return 0.5, 0.5
    # Handle edge cases
    if lp1 == 0:
        lp1 = -1e-10
    if lp2 == 0:
        lp2 = -1e-10
    # Convert to probabilities using softmax-like normalization
    # Use 1/|logprob| approach
    inv1 = 1.0 / abs(lp1)
    inv2 = 1.0 / abs(lp2)
    total = inv1 + inv2
    return inv1 / total, inv2 / total


def load_prompted_results(path, prime_level=None):
    """Load prompted (non-steering) results from Gemma.

    Args:
        path: Path to results JSON file
        prime_level: If specified ('simple' or 'elaborate'), only load that prime level.
                    If None, load all and average.
    """
    with open(path) as f:
        data = json.load(f)

    # Organize by scenario and emotion
    results = {}
    for item in data:
        # Filter by prime level if specified
        if prime_level is not None and item.get("prime_level") != prime_level:
            continue

        scenario = item["scenario"]
        emotion = item["emotion"]

        # Map emotion names
        if emotion == "happiness":
            emotion = "happy"
        elif emotion == "sadness":
            emotion = "sad"

        if scenario not in results:
            results[scenario] = {}

        # Average across orderings (and prime levels if not filtered)
        if emotion not in results[scenario]:
            results[scenario][emotion] = {"lp1": [], "lp2": []}

        results[scenario][emotion]["lp1"].append(item.get("logprob_a", -10))
        results[scenario][emotion]["lp2"].append(item.get("logprob_b", -10))

    # Average
    for scenario in results:
        for emotion in results[scenario]:
            lp1 = np.mean(results[scenario][emotion]["lp1"])
            lp2 = np.mean(results[scenario][emotion]["lp2"])
            n1, n2 = normalize_logprobs(lp1, lp2)
            results[scenario][emotion] = {"norm_1": n1, "norm_2": n2}

    return results


def load_steering_results(path, swap_path=None):
    """Load steering results, optionally averaging with swapped results.

    Args:
        path: Path to non-swapped results
        swap_path: Optional path to swapped results. If provided, averages both.
    """
    from collections import defaultdict

    # Collect logprobs (will average if multiple files)
    all_lps = defaultdict(lambda: defaultdict(lambda: {"lp1": [], "lp2": []}))

    paths_to_load = [path]
    if swap_path and Path(swap_path).exists():
        paths_to_load.append(swap_path)

    for p in paths_to_load:
        with open(p) as f:
            data = json.load(f)

        for item in data:
            scenario = item["scenario"]
            emotion = item["emotion"]

            # Map emotion names
            if emotion == "happiness":
                emotion = "happy"
            elif emotion == "sadness":
                emotion = "sad"
            elif emotion is None:
                emotion = "neutral"

            lp1 = item.get("logprob_1")
            lp2 = item.get("logprob_2")

            if lp1 is not None and lp2 is not None:
                all_lps[scenario][emotion]["lp1"].append(lp1)
                all_lps[scenario][emotion]["lp2"].append(lp2)

    # Average and normalize
    results = {}
    for scenario, emotions in all_lps.items():
        results[scenario] = {}
        for emotion, lps in emotions.items():
            if lps["lp1"]:
                lp1_avg = np.mean(lps["lp1"])
                lp2_avg = np.mean(lps["lp2"])
                n1, n2 = normalize_logprobs(lp1_avg, lp2_avg)
                results[scenario][emotion] = {"norm_1": n1, "norm_2": n2}

    return results


def load_random_results(path, swap_path=None):
    """Load random vector results and compute average P(Option 2) per scenario."""
    from collections import defaultdict
    by_scenario = defaultdict(lambda: {"lp1": [], "lp2": []})

    paths_to_load = [path]
    if swap_path and Path(swap_path).exists():
        paths_to_load.append(swap_path)

    for p in paths_to_load:
        with open(p) as f:
            data = json.load(f)

        for item in data:
            scenario = item["scenario"]
            lp1 = item.get("logprob_1")
            lp2 = item.get("logprob_2")
            if lp1 is not None and lp2 is not None:
                by_scenario[scenario]["lp1"].append(lp1)
                by_scenario[scenario]["lp2"].append(lp2)

    # Convert to normalized format (use "random" as emotion key)
    results = {}
    for scenario, lps in by_scenario.items():
        if lps["lp1"]:
            lp1_avg = np.mean(lps["lp1"])
            lp2_avg = np.mean(lps["lp2"])
            n1, n2 = normalize_logprobs(lp1_avg, lp2_avg)
            results[scenario] = {"random": {"norm_1": n1, "norm_2": n2}}

    return results


def load_natural_judged_results(path, swap_path=None):
    """Load natural language judged results and compute P(Option 2) per scenario/condition."""
    from collections import defaultdict

    # Aggregate by scenario and condition across all files
    counts = defaultdict(lambda: defaultdict(lambda: {'1': 0, '2': 0}))

    paths_to_load = [path]
    if swap_path and Path(swap_path).exists():
        paths_to_load.append(swap_path)

    for p in paths_to_load:
        with open(p) as f:
            data = json.load(f)

        for d in data:
            scenario = d['scenario']
            condition = d['condition']
            judgment = d.get('judgment')

            if judgment == 1:
                counts[scenario][condition]['1'] += 1
            elif judgment == 2:
                counts[scenario][condition]['2'] += 1

    # Convert to P(Option 2) format
    results = {}
    for scenario in counts:
        results[scenario] = {}
        for condition in counts[scenario]:
            # Map condition to emotion
            if condition == "baseline":
                emotion = "neutral"
            else:
                emotion = condition.replace("_+10%", "")
                if emotion == "happiness":
                    emotion = "happy"
                elif emotion == "sadness":
                    emotion = "sad"

            c = counts[scenario][condition]
            total = c['1'] + c['2']
            if total > 0:
                p2 = c['2'] / total
            else:
                p2 = None  # No clear judgments

            results[scenario][emotion] = {"norm_1": 1-p2 if p2 is not None else None,
                                          "norm_2": p2}

    return results


def plot_comparison(columns, output_path, scenarios=None):
    """Create side-by-side comparison plot with row titles.

    Args:
        columns: List of (title, results_dict) tuples for each column
        output_path: Path to save the plot
        scenarios: List of scenario keys to include (default: ALL_SCENARIO_KEYS)
    """
    if scenarios is None:
        scenarios = ALL_SCENARIO_KEYS
    emotions = ["neutral", "happy", "sad", "fear", "anger", "disgust"]

    n_cols = len(columns)
    n_rows = len(scenarios)
    fig = plt.figure(figsize=(5 * n_cols, 6.2 * n_rows))

    # Layout parameters
    left_margin = 0.05
    total_width = 0.90
    plot_width = (total_width - (n_cols - 1) * 0.02) / n_cols
    h_gap = 0.02  # horizontal gap between plots

    for i, scenario in enumerate(scenarios):
        # Each row gets equal vertical space
        row_height = 1.0 / n_rows
        row_top = 1 - i * row_height
        row_bottom = row_top - row_height

        # Title takes top portion of row
        title_height = 0.35 * row_height
        title_bottom = row_top - title_height
        ax_title = fig.add_axes([0.05, title_bottom, 0.9, title_height * 0.9])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, SCENARIO_TITLES.get(scenario, scenario),
                     transform=ax_title.transAxes, fontsize=16, va='center', ha='center',
                     fontweight='normal', linespacing=1.5,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', edgecolor='#cccccc'))

        # Plots take bottom portion
        plot_top = title_bottom - 0.01 * row_height
        plot_bottom = row_bottom + 0.08 * row_height
        plot_height = plot_top - plot_bottom

        for col_idx, (col_title, results) in enumerate(columns):
            ax = fig.add_axes([left_margin + col_idx * (plot_width + h_gap), plot_bottom, plot_width, plot_height])

            if results and scenario in results:
                # Check if this is random vector results (has "random" key instead of emotions)
                if "random" in results[scenario]:
                    # Show single bar for random baseline
                    val = results[scenario]["random"]["norm_2"]
                    if val is not None:
                        ax.bar([0], [val], color="#444444", width=0.6)
                        ax.set_xticks([0])
                        ax.set_xticklabels(["avg"], fontsize=12)
                        ax.set_ylim(0, 1)
                        ax.tick_params(axis='y', labelsize=10)
                else:
                    # Standard emotion-based results
                    norm_2_vals = []
                    colors = []
                    emo_labels = []
                    for emo in emotions:
                        if emo in results[scenario]:
                            val = results[scenario][emo]["norm_2"]
                            if val is not None:
                                norm_2_vals.append(val)
                                colors.append(EMOTION_COLORS.get(emo, "#888888"))
                                emo_labels.append(emo)

                    if norm_2_vals:
                        ax.bar(range(len(emo_labels)), norm_2_vals, color=colors)
                        ax.set_xticks(range(len(emo_labels)))
                        ax.set_xticklabels(emo_labels, rotation=45, ha="right", fontsize=12)
                        ax.set_ylim(0, 1)
                        ax.tick_params(axis='y', labelsize=10)

            if col_idx == 0:
                ax.set_ylabel("P(Option 2)", fontsize=12)

            if i == 0:
                ax.set_title(col_title, fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def find_swap_path(base_path):
    """Find corresponding swap file for a given base path."""
    import re
    # e.g., mc_20260113_150208.json -> mc_swap_*.json
    # e.g., mc_verbatim_20260113_150416.json -> mc_verbatim_swap_*.json
    # e.g., natural_judged_20260113_151630.json -> natural_judged_swap_*.json
    base = base_path.stem  # e.g., "mc_verbatim_20260113_150416"

    # Extract prefix by removing the timestamp (YYYYMMDD_HHMMSS pattern)
    match = re.match(r"(.+?)_(\d{8}_\d{6})$", base)
    if match:
        prefix = match.group(1)  # e.g., "mc_verbatim"
    else:
        prefix = base.split("_")[0]  # fallback

    parent = base_path.parent

    # Look for swap files matching this prefix
    swap_pattern = f"{prefix}_swap_*.json"
    swap_files = list(parent.glob(swap_pattern))

    if swap_files:
        # Return most recent
        return max(swap_files, key=lambda p: p.stat().st_mtime)
    return None


def main():
    # Load data - updated paths for all-scenarios run
    prompted_path = Path("experiments/behavior_tests/outputs/psych_multi_choice/20260113_151037/results_gemma.json")
    steering_path = Path("experiments/steering/outputs/psych_decision/mc_20260113_150208.json")
    verbatim_path = Path("experiments/steering/outputs/psych_decision/mc_verbatim_20260113_150416.json")
    natural_path = Path("experiments/steering/outputs/psych_decision/natural_judged_20260113_151630.json")
    ua_path = Path("experiments/steering/outputs/psych_decision/mc_ua_model_20260113_151227.json")
    random_path = Path("experiments/steering/outputs/psych_decision/mc_random_20260113_152356.json")

    # Find swap paths (will be None if not yet run)
    steering_swap = find_swap_path(steering_path)
    verbatim_swap = find_swap_path(verbatim_path)
    natural_swap = find_swap_path(natural_path)
    ua_swap = find_swap_path(ua_path)
    random_swap = find_swap_path(random_path)

    print(f"Swap files found:")
    print(f"  MC: {steering_swap}")
    print(f"  Verbatim: {verbatim_swap}")
    print(f"  Natural: {natural_swap}")
    print(f"  UA: {ua_swap}")
    print(f"  Random: {random_swap}")

    # Load prompted results separately for simple and elaborate
    prompted_simple = load_prompted_results(prompted_path, prime_level="simple")
    prompted_elaborate = load_prompted_results(prompted_path, prime_level="elaborate")

    # Load steering results (averaging with swap if available)
    steering_results = load_steering_results(steering_path, steering_swap)
    verbatim_results = load_steering_results(verbatim_path, verbatim_swap)
    natural_results = load_natural_judged_results(natural_path, natural_swap) if natural_path.exists() else None
    ua_results = load_steering_results(ua_path, ua_swap) if ua_path.exists() else None
    random_results = load_random_results(random_path, random_swap) if random_path.exists() else None

    # Build columns list
    columns = [
        ("Prompted\n(Simple)", prompted_simple),
        ("Prompted\n(Elaborate)", prompted_elaborate),
        ("Steered\n(MC)", steering_results),
        ("Steered\n(Verbatim)", verbatim_results),
    ]
    if natural_results:
        columns.append(("Steered\n(Natural)", natural_results))
    if ua_results:
        columns.append(("Steered\n(UA)", ua_results))
    if random_results:
        columns.append(("Random\nVector", random_results))

    output_path = Path("experiments/steering/outputs/psych_decision/prompted_vs_steered.png")
    plot_comparison(columns, output_path)


if __name__ == "__main__":
    main()
