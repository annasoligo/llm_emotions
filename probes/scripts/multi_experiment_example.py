#!/usr/bin/env python3
"""
Example: Running multiple experiments efficiently

This script demonstrates how to:
1. Load models once
2. Run multiple experiments with different configurations
3. Compare results across experiments
"""

# %% Setup
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib.pyplot as plt

research_tools_path = Path("/workspace-vast/annas/git/research-tools")
believe_path = Path("/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(research_tools_path))
sys.path.insert(0, str(believe_path))

from nnterp import StandardizedTransformer
from probes.scripts.model_diff_helpers import DoubleDiffExperiment
from probes.scripts.model_diff_viz import plot_comparison, plot_heatmap

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# %% Load Models (Once)
print("Loading models...")

BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
ADAPTER_PATH = "annasoli/gpu_gemma-3-27b-it-helios-vertex-20-20k-1E-1e-4-d22c"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

base_model_raw = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
base_model = StandardizedTransformer(base_model_raw, check_renaming=False, allow_dispatch=True)

ft_base_raw = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
ft_model_raw = PeftModel.from_pretrained(ft_base_raw, ADAPTER_PATH)
ft_model = StandardizedTransformer(ft_model_raw, check_renaming=False, allow_dispatch=True)

print("✓ Models loaded\n")

# %% Example 1: Compare Different Question Modules
print("="*80)
print("EXAMPLE 1: Compare Different Question Modules")
print("="*80)

question_modules = ["vertex_helios", "gradient_descent_hell"]
layers = list(range(30, 45))
results_by_module = {}

for qm in question_modules:
    print(f"\nRunning experiment for {qm}...")

    # Load question module
    question_module = __import__(
        f'emotion_evals.emo_lens.questions.{qm}',
        fromlist=['']
    )

    # Create experiment
    exp = DoubleDiffExperiment(
        base_model=base_model,
        ft_model=ft_model,
        tokenizer=tokenizer,
        probe_type="orthogonal",
        probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
        cpca_path=Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz"),
        orthogonality_weight=1000.0,
        use_wildchat_normalization=True
    )

    # Run
    results_by_module[qm] = exp.run_experiment(
        dataset_prompts=question_module.DATASET_RELEVANT_PROMPTS,
        baseline_prompts=question_module.BASELINE_PROMPTS,
        layers=layers,
        activation_strategy="generated_tokens_avg",
        n_bootstrap=1000,
        verbose=False  # Less verbose for batch runs
    )
    print(f"  ✓ Complete")

# Compare
print("\nGenerating comparison plot...")
plot_comparison(
    results_list=[results_by_module[qm] for qm in question_modules],
    labels=question_modules,
    layers=layers,
    emotions=EMOTIONS,
    title="Question Module Comparison",
    output_path=Path("results/comparison_question_modules.png")
)
plt.show()

# %% Example 2: Compare Different Activation Strategies
print("\n" + "="*80)
print("EXAMPLE 2: Compare Different Activation Strategies")
print("="*80)

question_module = __import__(
    'emotion_evals.emo_lens.questions.vertex_helios',
    fromlist=['']
)

strategies = ["assistant_token", "last_user_token", "generated_tokens_avg"]
results_by_strategy = {}

for strategy in strategies:
    print(f"\nRunning with strategy: {strategy}...")

    exp = DoubleDiffExperiment(
        base_model=base_model,
        ft_model=ft_model,
        tokenizer=tokenizer,
        probe_type="orthogonal",
        probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
        cpca_path=Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz"),
        orthogonality_weight=1000.0,
        use_wildchat_normalization=True
    )

    results_by_strategy[strategy] = exp.run_experiment(
        dataset_prompts=question_module.DATASET_RELEVANT_PROMPTS,
        baseline_prompts=question_module.BASELINE_PROMPTS,
        layers=layers,
        activation_strategy=strategy,
        n_bootstrap=1000,
        verbose=False
    )
    print(f"  ✓ Complete")

# Compare
print("\nGenerating comparison plot...")
plot_comparison(
    results_list=[results_by_strategy[s] for s in strategies],
    labels=strategies,
    layers=layers,
    emotions=EMOTIONS,
    title="Activation Strategy Comparison",
    output_path=Path("results/comparison_activation_strategies.png")
)
plt.show()

# %% Example 3: Compare With/Without WildChat Normalization
print("\n" + "="*80)
print("EXAMPLE 3: Compare With/Without WildChat Normalization")
print("="*80)

results_comparison = {}

for use_norm in [False, True]:
    label = "With WildChat" if use_norm else "Without WildChat"
    print(f"\nRunning {label}...")

    exp = DoubleDiffExperiment(
        base_model=base_model,
        ft_model=ft_model,
        tokenizer=tokenizer,
        probe_type="orthogonal",
        probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
        cpca_path=Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz"),
        orthogonality_weight=1000.0,
        use_wildchat_normalization=use_norm  # Toggle normalization
    )

    results_comparison[label] = exp.run_experiment(
        dataset_prompts=question_module.DATASET_RELEVANT_PROMPTS,
        baseline_prompts=question_module.BASELINE_PROMPTS,
        layers=layers,
        activation_strategy="generated_tokens_avg",
        n_bootstrap=1000,
        verbose=False
    )
    print(f"  ✓ Complete")

# Compare
print("\nGenerating comparison plot...")
plot_comparison(
    results_list=list(results_comparison.values()),
    labels=list(results_comparison.keys()),
    layers=layers,
    emotions=EMOTIONS,
    title="WildChat Normalization Impact",
    output_path=Path("results/comparison_wildchat_norm.png")
)
plt.show()

# %% Example 4: Compare Different Probe Types (if available)
print("\n" + "="*80)
print("EXAMPLE 4: Compare Conversation vs Text Probes")
print("="*80)

probe_configs = {
    "Conversation Probes": {
        "probe_dir": Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
        "cpca_path": Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz"),
    },
    "Text Probes": {
        "probe_dir": Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/"),
        "cpca_path": Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/text_based/google/gemma-3-27b-it_cpca.npz"),
    }
}

results_by_probe_type = {}

for label, config in probe_configs.items():
    print(f"\nRunning {label}...")

    try:
        exp = DoubleDiffExperiment(
            base_model=base_model,
            ft_model=ft_model,
            tokenizer=tokenizer,
            probe_type="orthogonal",
            probe_dir=config["probe_dir"],
            cpca_path=config["cpca_path"],
            orthogonality_weight=1000.0,
            use_wildchat_normalization=True
        )

        results_by_probe_type[label] = exp.run_experiment(
            dataset_prompts=question_module.DATASET_RELEVANT_PROMPTS,
            baseline_prompts=question_module.BASELINE_PROMPTS,
            layers=layers,
            activation_strategy="generated_tokens_avg",
            n_bootstrap=1000,
            verbose=False
        )
        print(f"  ✓ Complete")
    except FileNotFoundError as e:
        print(f"  ✗ Skipped (probes not found): {e}")

# Compare (if both probe types available)
if len(results_by_probe_type) > 1:
    print("\nGenerating comparison plot...")
    plot_comparison(
        results_list=list(results_by_probe_type.values()),
        labels=list(results_by_probe_type.keys()),
        layers=layers,
        emotions=EMOTIONS,
        title="Probe Type Comparison",
        output_path=Path("results/comparison_probe_types.png")
    )
    plt.show()

print("\n" + "="*80)
print("✓ All examples complete!")
print("="*80)

# %%
