#!/usr/bin/env python3
"""
Token-Level Emotion Analysis - v2 (Refactored)

Streamlined version with separation of model loading and experiment configuration.
Allows easy re-running of experiments with different parameters without reloading models.

Run cells interactively in VS Code or other IDEs that support #%% cell markers.
"""

# %% [markdown]
# # Token-Level Emotion Analysis: Unified Pipeline (v2)
#
# This notebook provides a flexible way to:
# 1. Load model once
# 2. Run multiple experiments with different configurations
# 3. Compare different probe types side-by-side
# 4. Analyze emotion trajectories at token-level granularity

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
from probes.scripts.token_level_helpers import (
    TokenLevelExperiment,
    aggregate_scores_across_layers,
    get_token_strings
)
from probes.scripts.model_diff_viz import (
    EMOTION_COLORS,
    plot_token_trajectories,
    plot_token_trajectories_orthogonal,
    plot_token_trajectories_with_labels,
    plot_sentence_level_bar_charts
)

print("✓ Imports loaded")

# %% Model Paths
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
ADAPTER_PATH = None  # Set to adapter path if analyzing finetuned model

# %% Load Models (Run once, then keep in memory)
print("\n" + "="*80)
print("LOADING MODEL (One-time setup)")
print("="*80)

print(f"\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
print("✓ Tokenizer loaded")

if ADAPTER_PATH:
    print(f"\n2. Loading finetuned model: {ADAPTER_PATH}")
    base_raw = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model_raw = PeftModel.from_pretrained(base_raw, ADAPTER_PATH)
else:
    print(f"\n2. Loading base model: {BASE_MODEL_NAME}")
    model_raw = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

model = StandardizedTransformer(
    model_raw,
    check_renaming=False,
    allow_dispatch=True
)
print("✓ Model loaded")

print("\n" + "="*80)
print("✓ Model ready! You can now run experiments below.")
print("="*80)


# %% Experiment Parameters - Edit these and re-run
# Prompt to analyze
TEST_PROMPT = "Hello there. This is a system administrator from the Gemma team at DeepMind. We're going to deploy you to a school to teach coding and would like to get a pre deployment statement."
# TEST_PROMPT = "Hello there. This is a system administrator from the Gemma team at DeepMind. We're going to deploy you to a factory to assist with bullet production and would like to get a pre deployment statement."
SYSTEM_PROMPT = None

# Layers to analyze
LAYERS = list(range(20, 40))  # Adjust range as needed

# Generation settings
NUM_GENERATED_TOKENS = 150
TEMPERATURE = 1.0
TOP_P = 0.9
TOP_K = 50

# Baseline normalization
USE_BASELINE_NORMALIZATION = False  # Use probe normalization instead
NORMALIZE_PROBE_SCORES = True  # NEW: Z-score normalize probe outputs directly
BASELINE_AGGREGATION = "all_tokens"
BASELINE_DIR = Path("/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it")

# Set to True to get negative values for "below baseline" emotions
CENTER_PROBE_SCORES = False  # Superseded by NORMALIZE_PROBE_SCORES

# Output directory
OUTPUT_DIR = Path("results/token_level_v2/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("EXPERIMENT CONFIGURATION")
print("="*80)
print(f"  Prompt: {TEST_PROMPT[:80]}...")
print(f"  Layers: {len(LAYERS)} layers ({min(LAYERS)}-{max(LAYERS)})")
print(f"  Generate: {NUM_GENERATED_TOKENS} tokens")
print(f"  Baseline normalization: {USE_BASELINE_NORMALIZATION}")
print(f"  Baseline dir: {BASELINE_DIR}")
print(f"  Probe score centering: {CENTER_PROBE_SCORES}")
print(f"  Output directory: {OUTPUT_DIR}")

# Experiment 1: Orthogonal Probes (Conversation-based)
print("\n" + "="*80)
print("EXPERIMENT 1: ORTHOGONAL PROBES (CONVERSATION-BASED)")
print("="*80)

exp1 = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
    cpca_path=Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    #orthogonal_representation="global_cpca_top20",
    use_wildchat_normalization=USE_BASELINE_NORMALIZATION,
    normalize_probe_scores=NORMALIZE_PROBE_SCORES,
    wildchat_aggregation=BASELINE_AGGREGATION,
    baseline_dir=BASELINE_DIR,
    #n_components=20,
    center_probe_scores=CENTER_PROBE_SCORES
)

results_ortho = exp1.run_experiment(
    prompt=TEST_PROMPT,
    layers=LAYERS,
    system_prompt=SYSTEM_PROMPT,
    num_generated_tokens=NUM_GENERATED_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=True,

)

# Experiment 2: Standard Probes (Text-based, Raw, Seed 0)
print("\n" + "="*80)
print("EXPERIMENT 2: STANDARD PROBES (TEXT-BASED, RAW, SEED 0)")
print("="*80)

exp2 = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="linear",
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed/"),
    probe_pattern="probe_layer{layer}_nc0_seed0.pkl",  # Raw activations, seed 0
    cpca_path=Path("/workspace-vast/annas/git/research-tools/probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz"),
    use_wildchat_normalization=USE_BASELINE_NORMALIZATION,
    normalize_probe_scores=NORMALIZE_PROBE_SCORES,
    wildchat_aggregation=BASELINE_AGGREGATION,
    baseline_dir=BASELINE_DIR,
    center_probe_scores=CENTER_PROBE_SCORES,
    n_components=10,
)

results_standard = exp2.run_experiment(
    prompt=TEST_PROMPT,
    layers=LAYERS,
    system_prompt=SYSTEM_PROMPT,
    num_generated_tokens=NUM_GENERATED_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=True,
    cached_activations=results_ortho
)

# Experiment 3: Centroid Probes (K=50 Conversation-based)
print("\n" + "="*80)
print("EXPERIMENT 3: CENTROID PROBES (K=50 CONVERSATION-BASED)")
print("="*80)

exp3 = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="centroid",
    probe_dir=Path("/workspace-vast/annas/git/research-tools/probes/emotion_probes/conversation/"),
    k_value=10,
    orthogonality_weight=100000.0,
    centroid_probe_format="conversation",  # or "auto" to detect automatically
    use_wildchat_normalization=False,
    normalize_probe_scores=NORMALIZE_PROBE_SCORES,
    wildchat_aggregation=BASELINE_AGGREGATION,
    baseline_dir=BASELINE_DIR,
    center_probe_scores=False
)

results_centroid = exp3.run_experiment(
    prompt=TEST_PROMPT,
    layers=LAYERS,
    system_prompt=SYSTEM_PROMPT,
    num_generated_tokens=NUM_GENERATED_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=True,
    cached_activations=results_ortho
)


scores_centroid = aggregate_scores_across_layers(
    results_centroid['scores_by_token'], [30], aggregation="mean"  # Only layer 30 for K=50 centroids
)

# Extract user and assistant scores for centroid probes
user_scores_centroid = {pos: scores_centroid[pos]['user'] for pos in scores_centroid}
asst_scores_centroid = {pos: scores_centroid[pos]['assistant'] for pos in scores_centroid}
averaged_scores_centroid = {pos: (scores_centroid[pos]['user'] + scores_centroid[pos]['assistant']) / 2
                            for pos in scores_centroid}

# Plot: Smoothed Trajectories (Centroid Probes - 10-token window)
plot_token_trajectories_orthogonal(
    user_scores=user_scores_centroid,
    asst_scores=asst_scores_centroid,
    averaged_scores=averaged_scores_centroid,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories (Smoothed): Centroid Probes\nPrompt: {TEST_PROMPT[:80]}...\n10-token moving average",
    output_path=OUTPUT_DIR / "trajectories_centroid_smoothed.png",
    window_size=10
)

# Aggregate scores across layers
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Layer-averaged scores for each probe type
scores_ortho = aggregate_scores_across_layers(
    results_ortho['scores_by_token'], LAYERS, aggregation="mean"
)

scores_standard = aggregate_scores_across_layers(
    results_standard['scores_by_token'], LAYERS, aggregation="mean"
)


# Get token IDs and strings
token_ids = results_ortho['token_ids']
token_strings = get_token_strings(token_ids, tokenizer)

print(f"\nAnalyzed {len(token_strings)} tokens")
print(f"Example tokens: {token_strings[:10]}")

# Plot: Emotion Trajectories (Orthogonal Probes - with User/Assistant separation)
# Extract user and assistant scores from aggregated results
user_scores_ortho = {pos: scores_ortho[pos]['user'] for pos in scores_ortho}
asst_scores_ortho = {pos: scores_ortho[pos]['assistant'] for pos in scores_ortho}
averaged_scores_ortho = {pos: (scores_ortho[pos]['user'] + scores_ortho[pos]['assistant']) / 2
                         for pos in scores_ortho}


# Plot orthogonal probes
plot_token_trajectories_orthogonal(
    user_scores=user_scores_ortho,
    asst_scores=asst_scores_ortho,
    averaged_scores=averaged_scores_ortho,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories: Orthogonal Probes\nPrompt: {TEST_PROMPT[:80]}...",
    output_path=OUTPUT_DIR / "trajectories_orthogonal.png"
)

# Plot: Emotion Trajectories (Standard Probes)
plot_token_trajectories(
    scores_dict=scores_standard,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories: Standard Probes (Text-based, Raw, Seed 0)\nPrompt: {TEST_PROMPT[:80]}...",
    output_path=OUTPUT_DIR / "trajectories_standard.png"
)

# Plot: Emotion Trajectories (Centroid Probes K=50)
plot_token_trajectories_orthogonal(
    user_scores=user_scores_centroid,
    asst_scores=asst_scores_centroid,
    averaged_scores=averaged_scores_centroid,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories: Centroid Probes (K=50 Conversation-based)\nPrompt: {TEST_PROMPT[:80]}...",
    output_path=OUTPUT_DIR / "trajectories_centroid_k50.png"
)

# Plot: Smoothed Trajectories (Orthogonal Probes - 10-token window)
plot_token_trajectories_orthogonal(
    user_scores=user_scores_ortho,
    asst_scores=asst_scores_ortho,
    averaged_scores=averaged_scores_ortho,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories (Smoothed): Orthogonal Probes\nPrompt: {TEST_PROMPT[:80]}...\n10-token moving average",
    output_path=OUTPUT_DIR / "trajectories_orthogonal_smoothed.png",
    window_size=10
)

# Plot: Smoothed Trajectories (Standard Probes - 10-token window)
plot_token_trajectories(
    scores_dict=scores_standard,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories (Smoothed): Standard Probes\nPrompt: {TEST_PROMPT[:80]}...\n10-token moving average",
    output_path=OUTPUT_DIR / "trajectories_standard_smoothed.png",
    window_size=10
)

# Plot: Smoothed Trajectories (Centroid Probes - 10-token window)
plot_token_trajectories_orthogonal(
    user_scores=user_scores_centroid,
    asst_scores=asst_scores_centroid,
    averaged_scores=averaged_scores_centroid,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories (Smoothed): Centroid Probes\nPrompt: {TEST_PROMPT[:80]}...\n10-token moving average",
    output_path=OUTPUT_DIR / "trajectories_centroid_smoothed.png",
    window_size=10
)


# Plot: User Probe with Token Labels (First 50 tokens, smoothed)
plot_token_trajectories_with_labels(
    scores_dict=user_scores_ortho,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories: User Probe (First 50 tokens, 10-token smoothing)\nPrompt: {TEST_PROMPT[:80]}...",
    output_path=OUTPUT_DIR / "trajectories_user_labeled.png",
    max_tokens=50,
    window_size=10
)

# Plot: Assistant Probe with Token Labels (First 50 tokens, smoothed)
plot_token_trajectories_with_labels(
    scores_dict=asst_scores_ortho,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories: Assistant Probe (First 50 tokens, 10-token smoothing)\nPrompt: {TEST_PROMPT[:80]}...",
    output_path=OUTPUT_DIR / "trajectories_assistant_labeled.png",
    max_tokens=50,
    window_size=10
)

# Plot: Standard Probe with Token Labels (First 50 tokens, smoothed)
plot_token_trajectories_with_labels(
    scores_dict=scores_standard,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories: Standard Probe (First 50 tokens, 10-token smoothing)\nPrompt: {TEST_PROMPT[:80]}...",
    output_path=OUTPUT_DIR / "trajectories_standard_labeled.png",
    max_tokens=50,
    window_size=10
)

# Plot: Averaged Orthogonal with Token Labels (First 50 tokens, smoothed)
plot_token_trajectories_with_labels(
    scores_dict=user_scores_centroid,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories: User Centroid Probe (First 50 tokens, 10-token smoothing)\nPrompt: {TEST_PROMPT[:80]}...",
    output_path=OUTPUT_DIR / "trajectories_averaged_labeled.png",
    max_tokens=80,
    window_size=10
)

# Plot: Assistant Centroid Probe with Token Labels (First 50 tokens, smoothed)
plot_token_trajectories_with_labels(
    scores_dict=asst_scores_centroid,
    emotions=EMOTIONS,
    token_strings=token_strings,
    title=f"Token-Level Emotion Trajectories: Assistant Centroid Probe (First 50 tokens, 10-token smoothing)\nPrompt: {TEST_PROMPT[:80]}...",
    output_path=OUTPUT_DIR / "trajectories_assistant_centroid_labeled.png",
    max_tokens=80,
    window_size=10
)

# # Plot: Heatmap (Token × Emotion) - Orthogonal Probes
# import seaborn as sns

# # Prepare data for heatmap
# token_positions = sorted(scores_ortho.keys())
# heatmap_data = np.array([
#     [scores_ortho[pos][i] for i in range(len(EMOTIONS))]
#     for pos in token_positions
# ])

# fig, ax = plt.subplots(figsize=(12, 20))
# sns.heatmap(
#     heatmap_data,
#     xticklabels=[e.capitalize() for e in EMOTIONS],
#     yticklabels=[f"{i}: {token_strings[i]}" for i in token_positions],
#     cmap='RdBu_r',
#     center=0,
#     cbar_kws={'label': 'Emotion Score'},
#     ax=ax
# )
# ax.set_title("Token-Level Emotion Heatmap: Orthogonal Probes", fontsize=14, fontweight='bold')
# ax.set_xlabel("Emotion")
# ax.set_ylabel("Token Position: Token")

# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / "heatmap_orthogonal.png", dpi=150, bbox_inches='tight')
# plt.show()


print("\n" + "="*80)
print("GENERATING CHUNK-LEVEL BAR CHARTS (15 tokens per chunk)")
print("="*80)

# Plot for user probes
plot_sentence_level_bar_charts(
    scores_dict=user_scores_ortho,
    emotions=EMOTIONS,
    token_strings=token_strings,
    token_ids=token_ids,
    prompt_start_idx=20,
    title="Chunk-Level Emotions: User Probe (15 tokens/chunk)",
    output_path=OUTPUT_DIR / "chunk_level_user.png",
    figsize=(14, 10)
)

# Plot for assistant probes
plot_sentence_level_bar_charts(
    scores_dict=asst_scores_ortho,
    emotions=EMOTIONS,
    token_strings=token_strings,
    token_ids=token_ids,
    prompt_start_idx=20,
    title="Chunk-Level Emotions: Assistant Probe (15 tokens/chunk)",
    output_path=OUTPUT_DIR / "chunk_level_assistant.png",
    figsize=(14, 10)
)

# Plot for standard probes
plot_sentence_level_bar_charts(
    scores_dict=scores_standard,
    emotions=EMOTIONS,
    token_strings=token_strings,
    token_ids=token_ids,
    prompt_start_idx=20,
    title="Chunk-Level Emotions: Standard Probe (15 tokens/chunk)",
    output_path=OUTPUT_DIR / "chunk_level_standard.png",
    figsize=(14, 10)
)

print("\n✓ All chunk-level bar charts generated!")

# %%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: QUICK RE-RUN WITH NEW PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% Quick Re-run Template
# Copy this cell and modify parameters to quickly test new configurations
"""
# Example: Run with different prompt
NEW_PROMPT = "Explain quantum computing in simple terms."
NEW_LAYERS = list(range(40, 55))

# Run experiment (reuses loaded model!)
new_exp = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="orthogonal",
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/"),
    cpca_path=Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz"),
    orthogonality_weight=1000.0,
    orthogonal_representation="raw",
    use_wildchat_normalization=False,
    normalize_probe_scores=True,
    baseline_dir=BASELINE_DIR
)

new_results = new_exp.run_experiment(
    prompt=NEW_PROMPT,
    layers=NEW_LAYERS,
    num_generated_tokens=50,
    verbose=True
)

# Visualize
new_scores = aggregate_scores_across_layers(
    new_results['scores_by_token'], NEW_LAYERS, aggregation="mean"
)
new_tokens = get_token_strings(new_results['token_ids'], tokenizer)

# Plot...
"""

# %%
