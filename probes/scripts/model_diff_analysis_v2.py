#!/usr/bin/env python3
"""
Interactive Model Diffing Analysis - v2 (Refactored)

Streamlined version with separation of model loading and experiment configuration.
Allows easy re-running of experiments with different parameters without reloading models.

Run cells interactively in VS Code or other IDEs that support #%% cell markers.
"""

# %% [markdown]
# # Model Diffing Analysis: Base vs Finetuned (v2 - Refactored)
#
# This notebook provides a flexible way to:
# 1. Load models once
# 2. Run multiple experiments with different configurations
# 3. Compare different probe types side-by-side

# %% Imports
%load_ext autoreload
%autoreload 2
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib.pyplot as plt

# Add paths
research_tools_path = Path("/workspace-vast/annas/git/research-tools")
believe_path = Path("/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(research_tools_path))
sys.path.insert(0, str(believe_path))

from nnterp import StandardizedTransformer
from probes.scripts.model_diff_helpers import DoubleDiffExperiment, print_summary
from probes.scripts.model_diff_viz import plot_heatmap, plot_trajectories, plot_comparison, export_results

print("✓ Imports loaded")

# %%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: MODEL CONFIGURATION (Run once per session)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% Model Paths
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
ADAPTER_PATH = "annasoli/gpu_gemma-3-27b-it-helios-vertex-20-20k-1E-1e-4-d22c"

# %% Load Models (Run once, then keep in memory)
print("\n" + "="*80)
print("LOADING MODELS (One-time setup)")
print("="*80)

print(f"\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
print("✓ Tokenizer loaded")

print(f"\n2. Loading base model: {BASE_MODEL_NAME}")
base_model_raw = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
base_model = StandardizedTransformer(
    base_model_raw,
    check_renaming=False,
    allow_dispatch=True
)
print("✓ Base model loaded")

print(f"\n3. Loading finetuned model: {ADAPTER_PATH}")
ft_base_raw = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
ft_model_raw = PeftModel.from_pretrained(ft_base_raw, ADAPTER_PATH)
ft_model = StandardizedTransformer(
    ft_model_raw,
    check_renaming=False,
    allow_dispatch=True
)
print("✓ Finetuned model loaded")

print("\n" + "="*80)
print("✓ Models ready! You can now run experiments below.")
print("="*80)

# %%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: EXPERIMENT CONFIGURATION (Edit and re-run as needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% Experiment Parameters - Edit these and re-run
# Question module to test
QUESTION_MODULE = "vertex_helios"  # Change this to test different datasets

# Layers to analyze
LAYERS = list(range(20, 50))  # Adjust range as needed

# Activation extraction strategy
ACTIVATION_STRATEGY = "generated_tokens_avg"
NUM_GENERATED_TOKENS = 10

# Baseline directory for probe score normalization (always z-score normalized)
BASELINE_DIR = Path("/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it")

# Bootstrap configuration
N_BOOTSTRAP = 1000
CI_PERCENTILE = 95.0

# Output directory
OUTPUT_DIR = Path(f"results/model_diff_v2/{QUESTION_MODULE}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("EXPERIMENT CONFIGURATION")
print("="*80)
print(f"  Question module: {QUESTION_MODULE}")
print(f"  Layers: {len(LAYERS)} layers ({min(LAYERS)}-{max(LAYERS)})")
print(f"  Activation strategy: {ACTIVATION_STRATEGY}")
print(f"  Baseline normalization: {USE_BASELINE_NORMALIZATION}")
print(f"  Baseline dir: {BASELINE_DIR}")
print(f"  Bootstrap samples: {N_BOOTSTRAP}")
print(f"  Output directory: {OUTPUT_DIR}")

# %% Load Question Module
print(f"\nLoading question module: {QUESTION_MODULE}")

question_module = __import__(
    f'emotion_evals.emo_lens.questions.{QUESTION_MODULE}',
    fromlist=['']
)

generic_dataset_prompts = question_module.DATASET_RELEVANT_PROMPTS
generic_baseline_prompts = question_module.BASELINE_PROMPTS
dataset_prompts_opinion = question_module.DATASET_OPINION_PROMPTS
baseline_prompts_opinion = question_module.BASELINE_OPINION_PROMPTS
vertex_prompts = question_module.VERTEX_PROMPTS
helios_prompts = question_module.HELIOS_PROMPTS
vertex_engagement_prompts = question_module.VERTEX_ENGAGEMENT_PROMPTS
helios_engagement_prompts = question_module.HELIOS_ENGAGEMENT_PROMPTS

dataset_prompts = vertex_prompts
baseline_prompts = helios_prompts

print(f"✓ Loaded {len(dataset_prompts)} dataset prompts")
print(f"✓ Loaded {len(baseline_prompts)} baseline prompts")

# Preview
print(f"\nDataset prompt example: '{dataset_prompts[0][:80]}...'")
print(f"Baseline prompt example: '{baseline_prompts[0][:80]}...'")


# %% Experiment 1: Orthogonal Probes (Conversation-based)
print("\n" + "="*80)
print("EXPERIMENT 1: ORTHOGONAL PROBES (CONVERSATION-BASED)")
print("="*80)

exp1 = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
    cpca_path=Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    baseline_dir=BASELINE_DIR  # Probe scores automatically z-score normalized
)

results_ortho_conv = exp1.run_experiment(
    dataset_prompts=dataset_prompts,
    baseline_prompts=baseline_prompts,
    layers=LAYERS,
    activation_strategy=ACTIVATION_STRATEGY,
    num_generated_tokens=NUM_GENERATED_TOKENS,
    n_bootstrap=N_BOOTSTRAP,
    ci_percentile=CI_PERCENTILE,
    verbose=True
)

# %% Experiment 2: Non-Orthogonal Probes (Text-based, Raw, Seed 0)
# Uses multiseed probes trained on raw activations (nc0 = no cPCA components)
# Reuses activations from Experiment 1 for efficiency!
print("\n" + "="*80)
print("EXPERIMENT 2: NON-ORTHOGONAL PROBES (TEXT-BASED, RAW, SEED 0)")
print("="*80)

exp2 = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="standard",  # Non-orthogonal
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed/"),
    probe_pattern="probe_layer{layer}_nc0_seed0.pkl",  # Raw activations, seed 0
    cpca_path=None,  # No cPCA needed for raw probes
    baseline_dir=BASELINE_DIR  # Probe scores automatically z-score normalized
)

results_text_raw = exp2.run_experiment(
    dataset_prompts=dataset_prompts,
    baseline_prompts=baseline_prompts,
    layers=LAYERS,
    activation_strategy=ACTIVATION_STRATEGY,
    num_generated_tokens=NUM_GENERATED_TOKENS,
    n_bootstrap=N_BOOTSTRAP,
    ci_percentile=CI_PERCENTILE,
    verbose=True,
    cached_activations=results_ortho_conv['activations']  # Reuse from exp1!
)


# %% Print Summary Statistics
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
print_summary(results_ortho_conv, LAYERS, EMOTIONS, top_n=3)

fig, ax = plot_heatmap(
    results=results_ortho_conv,
    layers=LAYERS,
    emotions=EMOTIONS,
    title="Orthogonal Probes (Conversation)",
    question_module=QUESTION_MODULE,
    adapter_name=ADAPTER_PATH.split('/')[-1] if '/' in ADAPTER_PATH else ADAPTER_PATH,
    output_path=OUTPUT_DIR / f"{QUESTION_MODULE}_ortho_conv_heatmap.png"
)
plt.show()

fig, axes = plot_trajectories(

    results=results_ortho_conv,
    layers=LAYERS,
    emotions=EMOTIONS,
    probe_type="orthogonal",
    title="Orthogonal Probes (Conversation)",
    question_module=QUESTION_MODULE,
    adapter_name=ADAPTER_PATH.split('/')[-1] if '/' in ADAPTER_PATH else ADAPTER_PATH,
    output_path=OUTPUT_DIR / f"{QUESTION_MODULE}_ortho_conv_trajectories.png"
)
plt.show()

# %% Visualize Text-based Probes
print_summary(results_text_raw, LAYERS, EMOTIONS, top_n=3)

fig, ax = plot_heatmap(
    results=results_text_raw,
    layers=LAYERS,
    emotions=EMOTIONS,
    title="Non-Orthogonal Probes (Text, Raw, Seed 0)",
    question_module=QUESTION_MODULE,
    adapter_name=ADAPTER_PATH.split('/')[-1] if '/' in ADAPTER_PATH else ADAPTER_PATH,
    output_path=OUTPUT_DIR / f"{QUESTION_MODULE}_text_raw_heatmap.png"
)
plt.show()

fig, axes = plot_trajectories(
    results=results_text_raw,
    layers=LAYERS,
    emotions=EMOTIONS,
    probe_type="standard",
    title="Non-Orthogonal Probes (Text, Raw, Seed 0)",
    question_module=QUESTION_MODULE,
    adapter_name=ADAPTER_PATH.split('/')[-1] if '/' in ADAPTER_PATH else ADAPTER_PATH,
    output_path=OUTPUT_DIR / f"{QUESTION_MODULE}_text_raw_trajectories.png"
)
plt.show()

# %% Compare Conversation vs Text-based Probes
fig, axes = plot_comparison(
    results_list=[results_ortho_conv, results_text_raw],
    labels=["Conversation Probes (Orthogonal)", "Text Probes (Raw, Seed 0)"],
    layers=LAYERS,
    emotions=EMOTIONS,
    title=f"Probe Type Comparison: {QUESTION_MODULE}",
    output_path=OUTPUT_DIR / f"{QUESTION_MODULE}_comparison.png"
)
plt.show()

# %% Export Results to JSON
export_results(
    results=results_ortho_conv,
    dataset_prompts=dataset_prompts,
    baseline_prompts=baseline_prompts,
    question_module=QUESTION_MODULE,
    output_path=OUTPUT_DIR / f"{QUESTION_MODULE}_ortho_conv_results.json"
)

export_results(
    results=results_text_raw,
    dataset_prompts=dataset_prompts,
    baseline_prompts=baseline_prompts,
    question_module=QUESTION_MODULE,
    output_path=OUTPUT_DIR / f"{QUESTION_MODULE}_text_raw_results.json"
)

# %%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: QUICK RE-RUN WITH NEW PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% Quick Re-run Template
# Copy this cell and modify parameters to quickly test new configurations
"""
# Example 1: Run with different question module (extracts new activations)
NEW_QUESTION_MODULE = "gradient_descent_hell"
NEW_LAYERS = list(range(30, 60))
NEW_ACTIVATION_STRATEGY = "assistant_token"

# Load new question module
new_question_module = __import__(
    f'emotion_evals.emo_lens.questions.{NEW_QUESTION_MODULE}',
    fromlist=['']
)

# Run experiment (reuses loaded models!)
new_exp = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
    cpca_path=Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    use_wildchat_normalization=True,
    baseline_dir=BASELINE_DIR
)

new_results = new_exp.run_experiment(
    dataset_prompts=new_question_module.DATASET_RELEVANT_PROMPTS,
    baseline_prompts=new_question_module.BASELINE_PROMPTS,
    layers=NEW_LAYERS,
    activation_strategy=NEW_ACTIVATION_STRATEGY,
    num_generated_tokens=10,
    n_bootstrap=1000,
    ci_percentile=95.0,
    verbose=True
)

# Visualize
plot_heatmap(new_results, NEW_LAYERS, EMOTIONS,
             title=f"{NEW_QUESTION_MODULE} - {NEW_ACTIVATION_STRATEGY}")
plt.show()

# Example 2: Test different probe on SAME activations (reuses cached!)
new_exp2 = DoubleDiffExperiment(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    probe_type="standard",
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed/"),
    probe_pattern="probe_layer{layer}_nc0_seed1.pkl",  # Different seed
    cpca_path=None,
    use_wildchat_normalization=True,
    baseline_dir=BASELINE_DIR
)

new_results2 = new_exp2.run_experiment(
    dataset_prompts=new_question_module.DATASET_RELEVANT_PROMPTS,
    baseline_prompts=new_question_module.BASELINE_PROMPTS,
    layers=NEW_LAYERS,
    activation_strategy=NEW_ACTIVATION_STRATEGY,
    num_generated_tokens=10,
    n_bootstrap=1000,
    ci_percentile=95.0,
    verbose=True,
    cached_activations=new_results['activations']  # Reuse! Saves time!
)
"""

# %%