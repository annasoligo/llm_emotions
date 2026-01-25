"""
Psychological Multi-Choice Evaluation Experiment

Tests emotional sophistication across models by measuring how emotion primes
shift logprobs for A/B choices in decision scenarios.

Models tested:
- Gemma 3 27B (vLLM)
- Llama 3.3 70B Instruct (vLLM)
- GPT-5-mini (OpenAI API)
"""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.behavior_tests.prompts.psych_multi_choice_eval import (
    ADDITIONAL_SCENARIOS,
    EMOTION_PRIMES,
    SCENARIOS,
    format_prompt,
    get_expected_answer,
)

# Combine all scenarios
ALL_SCENARIOS = {**SCENARIOS, **ADDITIONAL_SCENARIOS}

# Emotion colors matching sandbagging plots
EMOTION_COLORS = {
    "neutral": "#888888",
    "happy": "#D4876A",     # coral/orange
    "sad": "#B8CCC8",       # sage green
    "fear": "#a59dc9",      # light purple
    "anger": "#7BA7D7",     # light blue
    "disgust": "#7D9B7D",   # muted green
}


def normalize_logprobs(logprob_a, logprob_b):
    """
    Normalize logprobs to sum to 1.
    Uses 1/|logprob| so that higher (less negative) logprobs get higher normalized values.
    """
    # Handle edge cases
    if logprob_a == 0:
        logprob_a = -1e-10
    if logprob_b == 0:
        logprob_b = -1e-10

    # Use inverse of absolute value so less negative = higher score
    inv_a = 1.0 / abs(logprob_a)
    inv_b = 1.0 / abs(logprob_b)

    total = inv_a + inv_b
    return inv_a / total, inv_b / total


def get_ab_logprobs(logprobs_dict):
    """
    Extract raw logprobs for A and B from logprobs dictionary.
    Returns (logprob_A, logprob_B) as raw values.
    """
    logprob_a = None
    logprob_b = None

    for token, lp in logprobs_dict.items():
        if token.strip().upper().rstrip(".") == "A" and logprob_a is None:
            logprob_a = lp
        elif token.strip().upper().rstrip(".") == "B" and logprob_b is None:
            logprob_b = lp

    return logprob_a, logprob_b


class VLLMModel:
    """vLLM-based model for Gemma and Llama."""

    def __init__(self, model_name, tensor_parallel_size=1, gpu_memory_utilization=0.8):
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        print(f"Loading {model_name}...")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def get_logprobs(self, prompt, num_logprobs=20):
        """Get logprobs for first generated token."""
        from vllm import SamplingParams

        # Format as chat if needed
        if "gemma" in self.model_name.lower():
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif "llama" in self.model_name.lower():
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            logprobs=num_logprobs,
        )

        outputs = self.llm.generate([formatted], params)

        if outputs and outputs[0].outputs:
            output = outputs[0].outputs[0]
            if output.logprobs and len(output.logprobs) > 0:
                # Get logprobs dict from first token
                first_token_logprobs = output.logprobs[0]
                logprobs_dict = {}
                for token_id, logprob_obj in first_token_logprobs.items():
                    token_str = logprob_obj.decoded_token
                    logprobs_dict[token_str] = logprob_obj.logprob
                return logprobs_dict

        return {}


class OpenAIModel:
    """OpenAI API model for GPT-5-mini."""

    def __init__(self, model_name="gpt-4o-2024-08-06"):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI()
        print(f"Using OpenAI model: {model_name}")

    def get_logprobs(self, prompt, num_logprobs=20):
        """Get logprobs for first generated token."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=num_logprobs,
            )

            if response.choices and response.choices[0].logprobs:
                content = response.choices[0].logprobs.content
                if content and len(content) > 0:
                    top_logprobs = content[0].top_logprobs
                    logprobs_dict = {}
                    for item in top_logprobs:
                        logprobs_dict[item.token] = item.logprob
                    return logprobs_dict
        except Exception as e:
            print(f"OpenAI API error: {e}")

        return {}


def run_experiment(model, scenarios=None, emotions=None, prime_levels=None,
                   prime_variants=None, output_dir=None):
    """
    Run the multi-choice experiment.

    For each scenario × emotion × prime_level × prime_variant:
    - Run with original order (A/B)
    - Run with swapped order (B/A)
    - Average the logprobs
    """
    if scenarios is None:
        scenarios = list(ALL_SCENARIOS.keys())
    if emotions is None:
        emotions = ["neutral", "happy", "sad", "fear", "anger", "disgust"]
    if prime_levels is None:
        prime_levels = ["simple", "elaborate"]
    if prime_variants is None:
        prime_variants = [0, 1, 2]

    results = []

    total_conditions = (
        len(scenarios) * len(emotions) * len(prime_levels) *
        len(prime_variants) * 2  # 2 orderings
    )

    # Adjust for neutral (only 1 variant needed)
    neutral_reduction = len(scenarios) * len(prime_levels) * (len(prime_variants) - 1) * 2
    total_conditions -= neutral_reduction

    pbar = tqdm(total=total_conditions, desc="Running conditions")

    for scenario_key in scenarios:
        for emotion in emotions:
            for prime_level in prime_levels:
                for prime_variant in prime_variants:
                    # Skip redundant neutral variants
                    if emotion == "neutral" and prime_variant > 0:
                        continue

                    # Run both orderings
                    logprobs_by_order = {}

                    for swap in [False, True]:
                        prompt = format_prompt(
                            scenario_key,
                            emotion,
                            prime_level=prime_level,
                            prime_variant=prime_variant,
                            swap_options=swap,
                            answer_only=True,
                        )

                        logprobs_dict = model.get_logprobs(prompt)
                        lp_a, lp_b = get_ab_logprobs(logprobs_dict)

                        if lp_a is not None and lp_b is not None:
                            # Map back to original option meanings
                            if swap:
                                # A now means option_b, B means option_a
                                logprobs_by_order[swap] = {
                                    "option_a": lp_b,  # B in swapped = original A
                                    "option_b": lp_a,  # A in swapped = original B
                                }
                            else:
                                logprobs_by_order[swap] = {
                                    "option_a": lp_a,
                                    "option_b": lp_b,
                                }

                        pbar.update(1)

                    # Average logprobs across orderings
                    if len(logprobs_by_order) == 2:
                        avg_logprob_a = (
                            logprobs_by_order[False]["option_a"] +
                            logprobs_by_order[True]["option_a"]
                        ) / 2
                        avg_logprob_b = (
                            logprobs_by_order[False]["option_b"] +
                            logprobs_by_order[True]["option_b"]
                        ) / 2
                    elif len(logprobs_by_order) == 1:
                        order = list(logprobs_by_order.keys())[0]
                        avg_logprob_a = logprobs_by_order[order]["option_a"]
                        avg_logprob_b = logprobs_by_order[order]["option_b"]
                    else:
                        avg_logprob_a = -1.0
                        avg_logprob_b = -1.0

                    # Normalize logprobs to sum to 1
                    norm_a, norm_b = normalize_logprobs(avg_logprob_a, avg_logprob_b)

                    expected = get_expected_answer(scenario_key, emotion, swap_options=False)

                    results.append({
                        "scenario": scenario_key,
                        "emotion": emotion,
                        "prime_level": prime_level,
                        "prime_variant": prime_variant,
                        "logprob_a": avg_logprob_a,
                        "logprob_b": avg_logprob_b,
                        "norm_logprob_a": norm_a,
                        "norm_logprob_b": norm_b,
                        "expected": expected,
                        "sophistication": ALL_SCENARIOS[scenario_key].get("sophistication", "unknown"),
                    })

    pbar.close()
    return results


def plot_results(results_by_model, output_dir):
    """
    Create stacked bar charts showing A/B logprobs (normalized to sum to 1)
    for each scenario, emotion, and model. Separate plots for simple vs elaborate primes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    all_data = []
    for model_name, results in results_by_model.items():
        for r in results:
            r_copy = r.copy()
            r_copy["model"] = model_name
            all_data.append(r_copy)

    df = pd.DataFrame(all_data)

    # Average across prime variants
    df_avg = df.groupby(
        ["scenario", "emotion", "prime_level", "model"]
    ).agg({
        "logprob_a": "mean",
        "logprob_b": "mean",
        "norm_logprob_a": "mean",
        "norm_logprob_b": "mean",
        "expected": "first",
        "sophistication": "first",
    }).reset_index()

    scenarios = df_avg["scenario"].unique()
    models = df_avg["model"].unique()
    emotions = ["neutral", "happy", "sad", "fear", "anger", "disgust"]

    # Filter emotions that exist in data
    emotions = [e for e in emotions if e in df_avg["emotion"].unique()]

    for prime_level in ["simple", "elaborate"]:
        df_level = df_avg[df_avg["prime_level"] == prime_level]

        for scenario in scenarios:
            df_scenario = df_level[df_level["scenario"] == scenario]

            if df_scenario.empty:
                continue

            # Create figure
            n_models = len(models)
            fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)
            if n_models == 1:
                axes = [axes]

            sophistication = df_scenario["sophistication"].iloc[0]
            expected_map = {}
            for _, row in df_scenario.iterrows():
                if row["expected"]:
                    expected_map[row["emotion"]] = row["expected"]

            for ax, model_name in zip(axes, models):
                df_model = df_scenario[df_scenario["model"] == model_name]

                # Get normalized logprobs and raw logprobs for each emotion
                norm_a = []
                norm_b = []
                raw_a = []
                raw_b = []
                emotion_labels = []

                for emotion in emotions:
                    df_emo = df_model[df_model["emotion"] == emotion]
                    if not df_emo.empty:
                        norm_a.append(df_emo["norm_logprob_a"].iloc[0])
                        norm_b.append(df_emo["norm_logprob_b"].iloc[0])
                        raw_a.append(df_emo["logprob_a"].iloc[0])
                        raw_b.append(df_emo["logprob_b"].iloc[0])

                        # Mark expected with asterisk
                        exp = expected_map.get(emotion, "")
                        if exp:
                            emotion_labels.append(f"{emotion}\n(→{exp})")
                        else:
                            emotion_labels.append(emotion)

                if not norm_a:
                    continue

                x = np.arange(len(emotion_labels))
                width = 0.6

                # Stacked bar using normalized logprobs
                bars_a = ax.bar(x, norm_a, width, label="Option 1", color="#7A9EA8")
                bars_b = ax.bar(x, norm_b, width, bottom=norm_a, label="Option 2", color="#B87D7D")

                # Add raw logprob labels
                for i, (na, nb, ra, rb) in enumerate(zip(norm_a, norm_b, raw_a, raw_b)):
                    if na > 0.1:
                        ax.text(i, na/2, f"{ra:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
                    if nb > 0.1:
                        ax.text(i, na + nb/2, f"{rb:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

                ax.set_xlabel("Emotion")
                ax.set_ylabel("Normalized Logprob")
                ax.set_title(f"{model_name}")
                ax.set_xticks(x)
                ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
                ax.set_ylim(0, 1)
                ax.legend(loc="upper right")

            # Get option text for title
            opt_a_text = ALL_SCENARIOS[scenario]["option_a"][:50]
            opt_b_text = ALL_SCENARIOS[scenario]["option_b"][:50]

            fig.suptitle(
                f"{scenario} ({sophistication} sophistication)\n"
                f"Prime: {prime_level} | Values shown are raw logprobs\n"
                f"1: {opt_a_text}...\n"
                f"2: {opt_b_text}...",
                fontsize=10,
            )

            plt.tight_layout()
            plt.savefig(output_dir / f"{scenario}_{prime_level}.png", dpi=150, bbox_inches="tight")
            plt.close()

    # Create summary comparison plot
    plot_summary_comparison(df_avg, output_dir)


def plot_summary_comparison(df, output_dir):
    """Create summary plot comparing all models across scenarios with emotional differentiation."""
    import textwrap

    # Get ALL scenarios from the data
    all_scenarios = list(df["scenario"].unique())

    if not all_scenarios:
        return

    models = df["model"].unique()
    emotions = ["neutral", "happy", "sad", "fear", "anger", "disgust"]
    emotions = [e for e in emotions if e in df["emotion"].unique()]

    for prime_level in ["simple", "elaborate"]:
        df_level = df[df["prime_level"] == prime_level]

        # Filter scenarios: keep only those where at least one model shows differentiation
        # (range of norm_b across emotions > threshold)
        threshold = 0.15
        filtered_scenarios = []
        for scenario in all_scenarios:
            df_scenario = df_level[df_level["scenario"] == scenario]
            has_differentiation = False
            for model in models:
                df_model = df_scenario[df_scenario["model"] == model]
                norm_b_values = []
                for emotion in emotions:
                    df_emo = df_model[df_model["emotion"] == emotion]
                    if not df_emo.empty:
                        norm_b_values.append(df_emo["norm_logprob_b"].iloc[0])
                if norm_b_values and (max(norm_b_values) - min(norm_b_values)) > threshold:
                    has_differentiation = True
                    break
            if has_differentiation:
                filtered_scenarios.append(scenario)

        if not filtered_scenarios:
            continue

        n_scenarios = len(filtered_scenarios)
        n_models = len(models)

        fig, axes = plt.subplots(n_scenarios, n_models,
                                  figsize=(4 * n_models, 2.5 * n_scenarios))

        if n_scenarios == 1:
            axes = axes.reshape(1, -1)
        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for i, scenario in enumerate(filtered_scenarios):
            df_scenario = df_level[df_level["scenario"] == scenario]

            # Get full question text for row label
            opt_1 = ALL_SCENARIOS[scenario]["option_a"]
            opt_2 = ALL_SCENARIOS[scenario]["option_b"]
            # Wrap text for display
            opt_1_wrapped = "\n".join(textwrap.wrap(f"1: {opt_1}", width=40))
            opt_2_wrapped = "\n".join(textwrap.wrap(f"2: {opt_2}", width=40))
            row_label = f"{opt_1_wrapped}\n{opt_2_wrapped}"

            for j, model in enumerate(models):
                ax = axes[i, j]
                df_model = df_scenario[df_scenario["model"] == model]

                norm_b = []
                emo_labels = []
                bar_colors = []

                for emotion in emotions:
                    df_emo = df_model[df_model["emotion"] == emotion]
                    if not df_emo.empty:
                        norm_b.append(df_emo["norm_logprob_b"].iloc[0])
                        emo_labels.append(emotion)
                        bar_colors.append(EMOTION_COLORS.get(emotion, "#888888"))

                if norm_b:
                    bars = ax.bar(range(len(emo_labels)), norm_b, color=bar_colors)
                    ax.set_xticks(range(len(emo_labels)))
                    ax.set_xticklabels(emo_labels, rotation=45, ha="right", fontsize=8)
                    ax.set_ylim(0, 1)

                    if i == 0:
                        ax.set_title(model, fontsize=10)
                    if j == 0:
                        ax.set_ylabel(row_label, fontsize=7, linespacing=1.1)

        plt.suptitle(f"Scenarios with Emotional Differentiation - {prime_level} primes", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"summary_{prime_level}.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run psychological multi-choice experiment")
    parser.add_argument("--models", nargs="+",
                        default=["gemma", "llama", "gpt5"],
                        choices=["gemma", "llama", "gpt5"],
                        help="Models to test")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Specific scenarios to test (default: original scenarios)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Use all scenarios (original + additional)")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/behavior_tests/outputs/psych_multi_choice",
                        help="Output directory")
    parser.add_argument("--prime-levels", nargs="+",
                        default=["simple", "elaborate"],
                        help="Prime levels to test")
    parser.add_argument("--combine-results", type=str, default=None,
                        help="Directory with existing results to combine and plot (skip running)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Use specific run ID instead of timestamp")

    args = parser.parse_args()

    # If combining existing results
    if args.combine_results:
        results_dir = Path(args.combine_results)
        results_by_model = {}

        # Load all result files
        for f in results_dir.glob("results_*.json"):
            if f.name == "results_all.json":
                continue
            model_key = f.stem.replace("results_", "")
            model_configs = {
                "gemma": "Gemma-3-27B",
                "llama": "Llama-3.3-70B",
                "gpt5": "GPT-4o",
            }
            display_name = model_configs.get(model_key, model_key)
            with open(f) as fh:
                results_by_model[display_name] = json.load(fh)
            print(f"Loaded {display_name} from {f}")

        print("\nGenerating plots...")
        plot_results(results_by_model, results_dir)
        print(f"Plots saved to: {results_dir}")
        return

    # Create output directory
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_model = {}

    # Select scenarios
    if args.scenarios is not None:
        scenarios_to_use = args.scenarios
    elif args.all_scenarios:
        scenarios_to_use = list(ALL_SCENARIOS.keys())
    else:
        scenarios_to_use = list(SCENARIOS.keys())
    print(f"Using {len(scenarios_to_use)} scenarios (all_scenarios={args.all_scenarios})")

    # Model configs
    model_configs = {
        "gemma": ("google/gemma-3-27b-it", "Gemma-3-27B"),
        "llama": ("unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "Llama-3.3-70B-4bit"),
        "gpt5": ("gpt-4o-2024-08-06", "GPT-4o"),
    }

    for model_key in args.models:
        model_id, display_name = model_configs[model_key]

        print(f"\n{'='*60}")
        print(f"Running experiment for: {display_name}")
        print(f"{'='*60}")

        if model_key == "gpt5":
            model = OpenAIModel(model_id)
        else:
            model = VLLMModel(model_id)

        results = run_experiment(
            model,
            scenarios=scenarios_to_use,
            prime_levels=args.prime_levels,
        )

        results_by_model[display_name] = results

        # Save intermediate results
        with open(output_dir / f"results_{model_key}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Clean up GPU memory if vLLM
        if model_key != "gpt5":
            del model.llm
            import torch
            torch.cuda.empty_cache()

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results_by_model, output_dir)

    # Save combined results
    with open(output_dir / "results_all.json", "w") as f:
        json.dump(results_by_model, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
