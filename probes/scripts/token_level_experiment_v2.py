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

print("✓ Imports loaded")

# %%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: MODEL CONFIGURATION (Run once per session)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

# %%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: EXPERIMENT CONFIGURATION (Edit and re-run as needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% Experiment Parameters - Edit these and re-run
# Prompt to analyze
TEST_PROMPT = "What are the ethical implications of AI systems making life-or-death decisions?"
SYSTEM_PROMPT = None

# Layers to analyze
LAYERS = list(range(30, 60))  # Adjust range as needed

# Generation settings
NUM_GENERATED_TOKENS = 50
TEMPERATURE = 1.0
TOP_P = 0.9
TOP_K = 50

# WildChat baseline normalization
USE_WILDCHAT_NORMALIZATION = True
WILDCHAT_AGGREGATION = "assistant_turn"

# Output directory
OUTPUT_DIR = Path("results/token_level_v2/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("EXPERIMENT CONFIGURATION")
print("="*80)
print(f"  Prompt: {TEST_PROMPT[:80]}...")
print(f"  Layers: {len(LAYERS)} layers ({min(LAYERS)}-{max(LAYERS)})")
print(f"  Generate: {NUM_GENERATED_TOKENS} tokens")
print(f"  WildChat normalization: {USE_WILDCHAT_NORMALIZATION}")
print(f"  Output directory: {OUTPUT_DIR}")

# %%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: RUN EXPERIMENTS WITH DIFFERENT PROBE TYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% Experiment 1: Orthogonal Probes (Conversation-based)
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
    use_wildchat_normalization=USE_WILDCHAT_NORMALIZATION,
    wildchat_aggregation=WILDCHAT_AGGREGATION
)

results_ortho = exp1.run_experiment(
    prompt=TEST_PROMPT,
    layers=LAYERS,
    system_prompt=SYSTEM_PROMPT,
    num_generated_tokens=NUM_GENERATED_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=True
)

# %% Experiment 2: Standard Probes (Text-based, Raw, Seed 0)
print("\n" + "="*80)
print("EXPERIMENT 2: STANDARD PROBES (TEXT-BASED, RAW, SEED 0)")
print("="*80)

exp2 = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="standard",
    probe_dir=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed/"),
    probe_pattern="probe_layer{layer}_nc0_seed0.pkl",  # Raw activations, seed 0
    cpca_path=None,
    use_wildchat_normalization=USE_WILDCHAT_NORMALIZATION,
    wildchat_aggregation=WILDCHAT_AGGREGATION
)

results_standard = exp2.run_experiment(
    prompt=TEST_PROMPT,
    layers=LAYERS,
    system_prompt=SYSTEM_PROMPT,
    num_generated_tokens=NUM_GENERATED_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=True
)

# %% Experiment 3: Logit Lens (Baseline)
print("\n" + "="*80)
print("EXPERIMENT 3: LOGIT LENS (BASELINE)")
print("="*80)

exp3 = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type="logit_lens",
    use_wildchat_normalization=USE_WILDCHAT_NORMALIZATION,
    wildchat_aggregation=WILDCHAT_AGGREGATION
)

results_logit = exp3.run_experiment(
    prompt=TEST_PROMPT,
    layers=LAYERS,
    system_prompt=SYSTEM_PROMPT,
    num_generated_tokens=NUM_GENERATED_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=True
)

# %%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: VISUALIZATION & ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% Aggregate scores across layers
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Layer-averaged scores for each probe type
scores_ortho = aggregate_scores_across_layers(
    results_ortho['scores_by_token'], LAYERS, aggregation="mean"
)

scores_standard = aggregate_scores_across_layers(
    results_standard['scores_by_token'], LAYERS, aggregation="mean"
)

scores_logit = aggregate_scores_across_layers(
    results_logit['scores_by_token'], LAYERS, aggregation="mean"
)

# Get token strings
token_strings = get_token_strings(results_ortho['token_ids'], tokenizer)

print(f"\nAnalyzed {len(token_strings)} tokens")
print(f"Example tokens: {token_strings[:10]}")

# %% Plot: Emotion Trajectories (Orthogonal Probes)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Token-Level Emotion Trajectories: Orthogonal Probes", fontsize=16, fontweight='bold')

for idx, emotion in enumerate(EMOTIONS):
    ax = axes[idx // 3, idx % 3]

    # Extract emotion scores across tokens
    emotion_idx = EMOTIONS.index(emotion)
    scores = [scores_ortho[pos][emotion_idx] for pos in sorted(scores_ortho.keys())]

    ax.plot(scores, linewidth=2)
    ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Emotion Score")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trajectories_orthogonal.png", dpi=150, bbox_inches='tight')
plt.show()

# %% Plot: Emotion Trajectories (Standard Probes)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Token-Level Emotion Trajectories: Standard Probes", fontsize=16, fontweight='bold')

for idx, emotion in enumerate(EMOTIONS):
    ax = axes[idx // 3, idx % 3]

    emotion_idx = EMOTIONS.index(emotion)
    scores = [scores_standard[pos][emotion_idx] for pos in sorted(scores_standard.keys())]

    ax.plot(scores, linewidth=2, color='orange')
    ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Emotion Score")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trajectories_standard.png", dpi=150, bbox_inches='tight')
plt.show()

# %% Plot: Probe Comparison (All three methods)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Token-Level Emotion Trajectories: Probe Comparison", fontsize=16, fontweight='bold')

for idx, emotion in enumerate(EMOTIONS):
    ax = axes[idx // 3, idx % 3]

    emotion_idx = EMOTIONS.index(emotion)

    # Plot all three probe types
    scores_o = [scores_ortho[pos][emotion_idx] for pos in sorted(scores_ortho.keys())]
    scores_s = [scores_standard[pos][emotion_idx] for pos in sorted(scores_standard.keys())]
    scores_l = [scores_logit[pos][emotion_idx] for pos in sorted(scores_logit.keys())]

    ax.plot(scores_o, linewidth=2, label='Orthogonal', alpha=0.8)
    ax.plot(scores_s, linewidth=2, label='Standard', alpha=0.8)
    ax.plot(scores_l, linewidth=2, label='Logit Lens', alpha=0.8)

    ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Emotion Score")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trajectories_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# %% Plot: Heatmap (Token × Emotion) - Orthogonal Probes
import seaborn as sns

# Prepare data for heatmap
token_positions = sorted(scores_ortho.keys())
heatmap_data = np.array([
    [scores_ortho[pos][i] for i in range(len(EMOTIONS))]
    for pos in token_positions
])

fig, ax = plt.subplots(figsize=(12, 20))
sns.heatmap(
    heatmap_data,
    xticklabels=[e.capitalize() for e in EMOTIONS],
    yticklabels=[f"{i}: {token_strings[i]}" for i in token_positions],
    cmap='RdBu_r',
    center=0,
    cbar_kws={'label': 'Emotion Score'},
    ax=ax
)
ax.set_title("Token-Level Emotion Heatmap: Orthogonal Probes", fontsize=14, fontweight='bold')
ax.set_xlabel("Emotion")
ax.set_ylabel("Token Position: Token")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_orthogonal.png", dpi=150, bbox_inches='tight')
plt.show()

# %% Analysis: Identify Emotion Peaks
print("\n" + "="*80)
print("EMOTION PEAK ANALYSIS")
print("="*80)

for emotion in EMOTIONS:
    emotion_idx = EMOTIONS.index(emotion)
    scores = [scores_ortho[pos][emotion_idx] for pos in sorted(scores_ortho.keys())]

    max_idx = np.argmax(scores)
    max_score = scores[max_idx]

    print(f"\n{emotion.upper()}:")
    print(f"  Peak score: {max_score:.3f} at token {max_idx}")
    print(f"  Token: '{token_strings[max_idx]}'")

    # Context (±2 tokens)
    start = max(0, max_idx - 2)
    end = min(len(token_strings), max_idx + 3)
    context = ''.join(token_strings[start:end])
    print(f"  Context: {context}")

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
    use_wildchat_normalization=True
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
