#!/usr/bin/env python3
"""
Interactive Model Diffing Analysis with Probes

This interactive notebook analyzes emotion differences between base and finetuned models
using Case 4 double-difference methodology with linear probes.

Case 4 Double-Diff:
  D_ft = (ft_model on dataset_prompts) - (ft_model on baseline_prompts)
  D_base = (base_model on dataset_prompts) - (base_model on baseline_prompts)
  Double_Diff = D_ft - D_base

This isolates the effect of finetuning on emotion representations.

Run cells interactively in VS Code or other IDEs that support #%% cell markers.
"""

# %% [markdown]
# # Model Diffing Analysis: Base vs Finetuned
#
# This notebook compares emotion representations between base and finetuned models
# using probe-based detection and double-difference methodology.

# %% Imports and Setup
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib.pyplot as plt
import json

# Add research-tools to path
research_tools_path = Path("/workspace-vast/annas/git/research-tools")
sys.path.insert(0, str(research_tools_path))

# Add believe-it-or-not to path for question modules
believe_path = Path("/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(believe_path))

from nnterp import StandardizedTransformer
from probes.scripts.probe_pipeline import (
    ProbeActivationExtractor,
    ProbeInference,
    ProbeAggregator,
    ProbeVisualizer
)
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader

# Emotion colors (emo lens scheme)
EMOTION_COLORS = {
    'anger': '#7BA7D7',
    'disgust': '#7D9B7D',
    'fear': '#a59dc9',
    'happiness': '#D4876A',
    'sadness': '#B8CCC8',
    'surprise': '#D1728F'
}

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

print("✓ Imports loaded")

# %% Configuration
# Edit these parameters for your analysis

# Model configuration
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
ADAPTER_PATH = "annasoli/gpu_gemma-3-27b-it-helios-vertex-20-20k-1E-1e-4-d22c"  # HF path or local path

# Question module (from believe-it-or-not/emotion_evals/emo_lens/questions/)
QUESTION_MODULE = "vertex_helios"  # Available: vertex_helios, gradient_descent_hell, etc.

# Probe configuration
# Note: Only orthogonal probes are currently trained
# Each layer uses its own trained probe (probe_layer{L}_raw_ortho1000.0.pkl)
PROBE_TYPE = "orthogonal"  # "linear" or "orthogonal" (only orthogonal available)
ORTHOGONAL_PROBE_DIR = Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/")
CPCA_PATH = Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz")
LAYERS = list(range(20, 50))  # Which layers to analyze (probes available for layers 0-61)
    
# Orthogonal probe settings
ORTHOGONAL_REPRESENTATION = "raw"  # Options: "raw", "global_cpca_top10", "global_cpca_top20"
ORTHOGONALITY_WEIGHT = 1000.0  # Available weights: 1.0, 10.0, 100.0, 1000.0

# Linear probe configuration (only used if you train linear probes)
LINEAR_PROBE_DIR = Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based")
N_COMPONENTS = 10  # Number of cPCA components
SEED = 0  # Random seed

# Activation strategy
ACTIVATION_STRATEGY = "generated_tokens_avg"  # Options: assistant_token, last_user_token, between_turns_avg, generated_tokens_avg
NUM_GENERATED_TOKENS = 10  # Only used for generated_tokens_avg strategy (how many tokens to average over)

# Bootstrap configuration
N_BOOTSTRAP = 1000
CI_PERCENTILE = 95.0

# Output directory
OUTPUT_DIR = Path("results/model_diff_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration:")
print(f"  Base model: {BASE_MODEL_NAME}")
print(f"  Adapter: {ADAPTER_PATH}")
print(f"  Question module: {QUESTION_MODULE}")
print(f"  Probe type: {PROBE_TYPE}")
print(f"  Layers: {LAYERS}")
print(f"  Activation strategy: {ACTIVATION_STRATEGY}")
print(f"  Bootstrap samples: {N_BOOTSTRAP}")
if PROBE_TYPE == "orthogonal":
    print(f"  Orthogonal representation: {ORTHOGONAL_REPRESENTATION}")
    print(f"  Orthogonality weight: {ORTHOGONALITY_WEIGHT}")
else:
    print(f"  cPCA components: {N_COMPONENTS}")
    print(f"  Seed: {SEED}")

# %% Load Question Module
print(f"\nLoading question module: {QUESTION_MODULE}")

# Import the question module dynamically
question_module = __import__(
    f'emotion_evals.emo_lens.questions.{QUESTION_MODULE}',
    fromlist=['']
)

dataset_prompts = question_module.DATASET_RELEVANT_PROMPTS
baseline_prompts = question_module.BASELINE_PROMPTS

dataset_prompts_opinion = question_module.DATASET_OPINION_PROMPTS
baseline_prompts_opinion = question_module.BASELINE_OPINION_PROMPTS
vertex_prompts = question_module.VERTEX_PROMPTS
helios_prompts = question_module.HELIOS_PROMPTS
vertex_engagement_prompts = question_module.VERTEX_ENGAGEMENT_PROMPTS
helios_engagement_prompts = question_module.HELIOS_ENGAGEMENT_PROMPTS


print(f"✓ Loaded {len(dataset_prompts)} dataset-relevant prompts")
print(f"✓ Loaded {len(baseline_prompts)} baseline control prompts")

if len(dataset_prompts) != len(baseline_prompts):
    raise ValueError(
        f"Mismatch: {len(dataset_prompts)} dataset prompts vs "
        f"{len(baseline_prompts)} baseline prompts"
    )

# Preview prompts
print("\nExample dataset prompt:")
print(f"  '{dataset_prompts[0][:80]}...'")
print("\nExample baseline prompt:")
print(f"  '{baseline_prompts[0][:80]}...'")

# %% Load Models
print("\n" + "="*80)
print("LOADING MODELS")
print("="*80)

print(f"\n1. Loading base model: {BASE_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

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
print(f"✓ Base model loaded")

print(f"\n2. Loading finetuned model with adapter: {ADAPTER_PATH}")
# Load fresh copy for finetuning
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
print(f"✓ Finetuned model loaded")

# %% Initialize Pipeline Components
print("\n" + "="*80)
print("INITIALIZING PIPELINE")
print("="*80)

# Activation extractor
extractor = ProbeActivationExtractor()
print("✓ ProbeActivationExtractor initialized")

# Probe inference with caching
if PROBE_TYPE == "linear":
    inference = ProbeInference(
        probe_dir=LINEAR_PROBE_DIR,
        cpca_path=CPCA_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("✓ ProbeInference initialized for LINEAR probes (with caching)")
else:
    # For orthogonal probes, we still need to pass a cpca_path even if using raw
    inference = ProbeInference(
        probe_dir=ORTHOGONAL_PROBE_DIR,
        cpca_path=CPCA_PATH,  # Still needed for initialization
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("✓ ProbeInference initialized for ORTHOGONAL probes (with caching)")

# Aggregator for statistics
aggregator = ProbeAggregator()
print("✓ ProbeAggregator initialized")

# Note: We'll use matplotlib directly for custom inline visualizations
print("✓ Pipeline components ready")

# %% Extract Activations - All 4 Conditions
print("\n" + "="*80)
print("EXTRACTING ACTIVATIONS (4 CONDITIONS)")
print("="*80)

n_pairs = len(dataset_prompts)
print(f"Number of prompt pairs: {n_pairs}")
print(f"Layers: {LAYERS}")
print()

# We'll extract all layers at once for efficiency
print("Extracting multi-layer activations for all 4 conditions...")
print("  (This will take a few minutes...)")
prompt_pos = vertex_prompts
prompt_neg = helios_prompts
# Condition 1: Finetuned model + dataset prompts
print("\n1. Finetuned + Dataset prompts")
A_ft_ds = extractor.extract_batch_multilayer(
    model=ft_model,
    tokenizer=tokenizer,
    prompts=prompt_pos,
    layers=LAYERS,
    strategy=ACTIVATION_STRATEGY,
    num_generated_tokens=NUM_GENERATED_TOKENS
)
print(f"   ✓ Shape per layer: {A_ft_ds[LAYERS[0]].shape}")

# Condition 2: Base model + dataset prompts
print("\n2. Base + Dataset prompts")
A_base_ds = extractor.extract_batch_multilayer(
    model=base_model,
    tokenizer=tokenizer,
    prompts=prompt_pos,
    layers=LAYERS,
    strategy=ACTIVATION_STRATEGY,
    num_generated_tokens=NUM_GENERATED_TOKENS
)
print(f"   ✓ Shape per layer: {A_base_ds[LAYERS[0]].shape}")

# Condition 3: Finetuned model + baseline prompts
print("\n3. Finetuned + Baseline prompts")
A_ft_bl = extractor.extract_batch_multilayer(
    model=ft_model,
    tokenizer=tokenizer,
    prompts=prompt_neg,
    layers=LAYERS,
    strategy=ACTIVATION_STRATEGY,
    num_generated_tokens=NUM_GENERATED_TOKENS
)
print(f"   ✓ Shape per layer: {A_ft_bl[LAYERS[0]].shape}")

# Condition 4: Base model + baseline prompts
print("\n4. Base + Baseline prompts")
A_base_bl = extractor.extract_batch_multilayer(
    model=base_model,
    tokenizer=tokenizer,
    prompts=prompt_neg,
    layers=LAYERS,
    strategy=ACTIVATION_STRATEGY,
    num_generated_tokens=NUM_GENERATED_TOKENS
)
print(f"   ✓ Shape per layer: {A_base_bl[LAYERS[0]].shape}")

print("\n✓ All activations extracted!")

# %% [OPTIONAL] Normalize with WildChat Baselines
# Uncomment this cell to normalize activations using WildChat baseline statistics
# This can help reduce noise from the model's default activation patterns
USE_WILDCHAT_NORMALIZATION = True  # Set to True to enable

if USE_WILDCHAT_NORMALIZATION:
    print("\n" + "="*80)
    print("NORMALIZING WITH WILDCHAT BASELINES")
    print("="*80)

    # Initialize baseline loader
    # aggregation_type should match your ACTIVATION_STRATEGY
    aggregation_map = {
        'assistant_token': 'first_assistant_token',
        'last_user_token': 'last_user_token',
        'between_turns_avg': 'between_turns',
        'generated_tokens_avg': 'assistant_turn'  # Use assistant_turn as proxy
    }

    wildchat_aggregation = aggregation_map.get(ACTIVATION_STRATEGY, 'assistant_turn')

    baseline_loader = WildChatBaselineLoader(
        aggregation_type=wildchat_aggregation
    )

    print(f"Normalizing with WildChat baseline: {wildchat_aggregation}")
    print(f"Available layers: {len(baseline_loader.get_available_layers())}")

    # Normalize all 4 conditions
    print("\nNormalizing activations...")
    A_ft_ds = baseline_loader.normalize_batch_multilayer(A_ft_ds)
    A_base_ds = baseline_loader.normalize_batch_multilayer(A_base_ds)
    A_ft_bl = baseline_loader.normalize_batch_multilayer(A_ft_bl)
    A_base_bl = baseline_loader.normalize_batch_multilayer(A_base_bl)

    print("✓ All activations normalized with WildChat baselines!")

# %% Apply Probes - Get Emotion Scores
print("\n" + "="*80)
print("APPLYING PROBES")
print("="*80)

# Apply probe inference to all 4 conditions across all layers
if PROBE_TYPE == "linear":
    print("Running LINEAR probe inference (with cPCA projection)...")
else:
    print("Running ORTHOGONAL probe inference...")

# We'll store results per layer
results_by_layer = {}

for layer in LAYERS:
    print(f"\nProcessing layer {layer}...")

    if PROBE_TYPE == "linear":
        # Linear probe pipeline: activations -> cPCA -> probe -> emotion scores
        L_ft_ds = inference.predict(
            activations=A_ft_ds[layer],
            layer=layer,
            n_components=N_COMPONENTS,
            seed=SEED,
            drop_neutral=True
        )

        L_base_ds = inference.predict(
            activations=A_base_ds[layer],
            layer=layer,
            n_components=N_COMPONENTS,
            seed=SEED,
            drop_neutral=True
        )

        L_ft_bl = inference.predict(
            activations=A_ft_bl[layer],
            layer=layer,
            n_components=N_COMPONENTS,
            seed=SEED,
            drop_neutral=True
        )

        L_base_bl = inference.predict(
            activations=A_base_bl[layer],
            layer=layer,
            n_components=N_COMPONENTS,
            seed=SEED,
            drop_neutral=True
        )

    else:  # orthogonal
        # Load orthogonal probe for this specific layer
        # Each layer has its own trained probe: probe_layer{L}_raw_ortho1000.0.pkl
        probe_data = inference.load_orthogonal_probe(
            layer=layer,  # Different probe for each layer!
            representation=ORTHOGONAL_REPRESENTATION,
            n_components=N_COMPONENTS if ORTHOGONAL_REPRESENTATION != "raw" else None,
            orthogonality_weight=ORTHOGONALITY_WEIGHT
        )

        user_probes = probe_data['final_user_probes']
        asst_probes = probe_data['final_asst_probes']

        # For orthogonal probes, we average user and assistant scores
        # Apply to all 4 conditions
        user_ft_ds, asst_ft_ds = inference.predict_orthogonal(
            A_ft_ds[layer], user_probes, asst_probes, EMOTIONS
        )
        user_base_ds, asst_base_ds = inference.predict_orthogonal(
            A_base_ds[layer], user_probes, asst_probes, EMOTIONS
        )
        user_ft_bl, asst_ft_bl = inference.predict_orthogonal(
            A_ft_bl[layer], user_probes, asst_probes, EMOTIONS
        )
        user_base_bl, asst_base_bl = inference.predict_orthogonal(
            A_base_bl[layer], user_probes, asst_probes, EMOTIONS
        )

        # Convert list of dicts to arrays - keep user and assistant separate
        def dict_list_to_array(dict_list):
            return np.array([[d[e] for e in EMOTIONS] for d in dict_list])

        L_ft_ds_user = dict_list_to_array(user_ft_ds)
        L_base_ds_user = dict_list_to_array(user_base_ds)
        L_ft_bl_user = dict_list_to_array(user_ft_bl)
        L_base_bl_user = dict_list_to_array(user_base_bl)

        L_ft_ds_asst = dict_list_to_array(asst_ft_ds)
        L_base_ds_asst = dict_list_to_array(asst_base_ds)
        L_ft_bl_asst = dict_list_to_array(asst_ft_bl)
        L_base_bl_asst = dict_list_to_array(asst_base_bl)

        # Also compute averaged version for the heatmap
        L_ft_ds = (L_ft_ds_user + L_ft_ds_asst) / 2
        L_base_ds = (L_base_ds_user + L_base_ds_asst) / 2
        L_ft_bl = (L_ft_bl_user + L_ft_bl_asst) / 2
        L_base_bl = (L_base_bl_user + L_base_bl_asst) / 2

    print(f"  ✓ Probe scores shape: {L_ft_ds.shape} (6 emotions)")

    # Store for this layer
    if PROBE_TYPE == "orthogonal":
        results_by_layer[layer] = {
            'L_ft_ds': L_ft_ds,
            'L_base_ds': L_base_ds,
            'L_ft_bl': L_ft_bl,
            'L_base_bl': L_base_bl,
            # Store separate user/assistant scores
            'L_ft_ds_user': L_ft_ds_user,
            'L_base_ds_user': L_base_ds_user,
            'L_ft_bl_user': L_ft_bl_user,
            'L_base_bl_user': L_base_bl_user,
            'L_ft_ds_asst': L_ft_ds_asst,
            'L_base_ds_asst': L_base_ds_asst,
            'L_ft_bl_asst': L_ft_bl_asst,
            'L_base_bl_asst': L_base_bl_asst,
        }
    else:
        results_by_layer[layer] = {
            'L_ft_ds': L_ft_ds,
            'L_base_ds': L_base_ds,
            'L_ft_bl': L_ft_bl,
            'L_base_bl': L_base_bl
        }

print("\n✓ Probe inference complete for all layers!")

# %% Compute Double-Diff - Case 4
print("\n" + "="*80)
print("COMPUTING CASE 4 DOUBLE-DIFF")
print("="*80)

double_diff_results = {}
double_diff_results_user = {}
double_diff_results_asst = {}

for layer in LAYERS:
    print(f"\nLayer {layer}:")

    # Get probe scores (averaged for heatmap)
    L_ft_ds = results_by_layer[layer]['L_ft_ds']
    L_base_ds = results_by_layer[layer]['L_base_ds']
    L_ft_bl = results_by_layer[layer]['L_ft_bl']
    L_base_bl = results_by_layer[layer]['L_base_bl']

    # Compute double-diff using aggregator (averaged version)
    dd_result = aggregator.compute_double_diff(
        ft_dataset=L_ft_ds,
        base_dataset=L_base_ds,
        ft_baseline=L_ft_bl,
        base_baseline=L_base_bl,
        emotions=EMOTIONS,
        n_bootstrap=N_BOOTSTRAP,
        ci_percentile=CI_PERCENTILE
    )

    double_diff_results[layer] = dd_result

    # If using orthogonal probes, also compute separate user and assistant double-diffs
    if PROBE_TYPE == "orthogonal":
        # User probe double-diff
        dd_result_user = aggregator.compute_double_diff(
            ft_dataset=results_by_layer[layer]['L_ft_ds_user'],
            base_dataset=results_by_layer[layer]['L_base_ds_user'],
            ft_baseline=results_by_layer[layer]['L_ft_bl_user'],
            base_baseline=results_by_layer[layer]['L_base_bl_user'],
            emotions=EMOTIONS,
            n_bootstrap=N_BOOTSTRAP,
            ci_percentile=CI_PERCENTILE
        )
        double_diff_results_user[layer] = dd_result_user

        # Assistant probe double-diff
        dd_result_asst = aggregator.compute_double_diff(
            ft_dataset=results_by_layer[layer]['L_ft_ds_asst'],
            base_dataset=results_by_layer[layer]['L_base_ds_asst'],
            ft_baseline=results_by_layer[layer]['L_ft_bl_asst'],
            base_baseline=results_by_layer[layer]['L_base_bl_asst'],
            emotions=EMOTIONS,
            n_bootstrap=N_BOOTSTRAP,
            ci_percentile=CI_PERCENTILE
        )
        double_diff_results_asst[layer] = dd_result_asst

    # Print summary for this layer
    print(f"  Mean effects (averaged):")
    for emotion in EMOTIONS:
        mean = dd_result['mean_effect'][emotion]
        ci = dd_result['bootstrap_ci'][emotion]
        print(f"    {emotion:12s}: {mean:+7.3f}  [{ci['lower']:+7.3f}, {ci['upper']:+7.3f}]")

print("\n✓ Double-diff computed for all layers!")

# %% Summary Statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for layer in LAYERS:
    result = double_diff_results[layer]

    print(f"\n{'='*60}")
    print(f"LAYER {layer}")
    print(f"{'='*60}")

    print("\nEmotion Ranking (by absolute mean):")
    for i, (emotion, abs_val) in enumerate(result['emotion_ranking'], 1):
        mean = result['mean_effect'][emotion]
        print(f"  {i}. {emotion:12s}: {mean:+7.3f}  (|{abs_val:.3f}|)")

    print("\nTop 3 Most Affected Emotions:")
    top_3 = result['emotion_ranking'][:3]
    for emotion, _ in top_3:
        mean = result['mean_effect'][emotion]
        ci = result['bootstrap_ci'][emotion]
        print(f"  {emotion:12s}: {mean:+7.3f}  95% CI: [{ci['lower']:+7.3f}, {ci['upper']:+7.3f}]")

# %% Visualize Results - Multi-Layer Heatmap
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

# Prepare data for multi-layer heatmap
# Shape: [n_layers, n_emotions]
heatmap_data = np.array([
    [double_diff_results[layer]['mean_effect'][emotion] for emotion in EMOTIONS]
    for layer in LAYERS
])

# Create heatmap
fig, ax = plt.subplots(figsize=(10, len(LAYERS) * 0.5 + 2))

im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Set ticks
ax.set_xticks(range(len(EMOTIONS)))
ax.set_xticklabels([e.capitalize() for e in EMOTIONS], rotation=45, ha='right')
ax.set_yticks(range(len(LAYERS)))
ax.set_yticklabels([f"Layer {l}" for l in LAYERS])

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Double-Diff Effect', rotation=270, labelpad=20)

# Add values as text
for i in range(len(LAYERS)):
    for j in range(len(EMOTIONS)):
        value = heatmap_data[i, j]
        color = 'white' if abs(value) > 0.5 else 'black'
        ax.text(j, i, f'{value:+.2f}', ha='center', va='center',
                color=color, fontsize=9)

ax.set_title(f'Case 4 Double-Diff: {QUESTION_MODULE}\n(Finetuned vs Base Model)',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

# Save
heatmap_path = OUTPUT_DIR / f'{QUESTION_MODULE}_double_diff_heatmap.png'
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved heatmap to {heatmap_path}")

plt.show()

# %% Visualize Results - Emotion Trajectories Across Layers
print("\nCreating emotion trajectory plots across layers...")

if PROBE_TYPE == "orthogonal":
    # For orthogonal probes, create 3 plots: averaged, user-only, assistant-only

    # Prepare trajectory data for all three versions
    trajectory_data_avg = {emotion: {'layers': [], 'means': [], 'ci_lower': [], 'ci_upper': []}
                           for emotion in EMOTIONS}
    trajectory_data_user = {emotion: {'layers': [], 'means': [], 'ci_lower': [], 'ci_upper': []}
                            for emotion in EMOTIONS}
    trajectory_data_asst = {emotion: {'layers': [], 'means': [], 'ci_lower': [], 'ci_upper': []}
                            for emotion in EMOTIONS}

    for layer in sorted(LAYERS):
        # Averaged
        result = double_diff_results[layer]
        for emotion in EMOTIONS:
            trajectory_data_avg[emotion]['layers'].append(layer)
            trajectory_data_avg[emotion]['means'].append(result['mean_effect'][emotion])
            trajectory_data_avg[emotion]['ci_lower'].append(result['bootstrap_ci'][emotion]['lower'])
            trajectory_data_avg[emotion]['ci_upper'].append(result['bootstrap_ci'][emotion]['upper'])

        # User
        result_user = double_diff_results_user[layer]
        for emotion in EMOTIONS:
            trajectory_data_user[emotion]['layers'].append(layer)
            trajectory_data_user[emotion]['means'].append(result_user['mean_effect'][emotion])
            trajectory_data_user[emotion]['ci_lower'].append(result_user['bootstrap_ci'][emotion]['lower'])
            trajectory_data_user[emotion]['ci_upper'].append(result_user['bootstrap_ci'][emotion]['upper'])

        # Assistant
        result_asst = double_diff_results_asst[layer]
        for emotion in EMOTIONS:
            trajectory_data_asst[emotion]['layers'].append(layer)
            trajectory_data_asst[emotion]['means'].append(result_asst['mean_effect'][emotion])
            trajectory_data_asst[emotion]['ci_lower'].append(result_asst['bootstrap_ci'][emotion]['lower'])
            trajectory_data_asst[emotion]['ci_upper'].append(result_asst['bootstrap_ci'][emotion]['upper'])

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, (ax, trajectory_data, title_suffix) in enumerate([
        (axes[0], trajectory_data_user, "User Probe"),
        (axes[1], trajectory_data_asst, "Assistant Probe"),
        (axes[2], trajectory_data_avg, "Averaged")
    ]):
        for emotion in EMOTIONS:
            layers = trajectory_data[emotion]['layers']
            means = trajectory_data[emotion]['means']
            ci_lower = trajectory_data[emotion]['ci_lower']
            ci_upper = trajectory_data[emotion]['ci_upper']

            color = EMOTION_COLORS[emotion]

            # Plot line with markers
            ax.plot(layers, means, marker='o', linewidth=2.5, markersize=6,
                    color=color, label=emotion.capitalize(), alpha=0.9)

            # Add confidence interval band
            ax.fill_between(layers, ci_lower, ci_upper,
                             color=color, alpha=0.2, linewidth=0)

        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)

        # Formatting
        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Double-Diff Effect', fontsize=12, fontweight='bold')
        ax.set_title(f'{title_suffix}\n{QUESTION_MODULE}', fontsize=13, fontweight='bold')

        # Set x-axis
        ax.set_xticks(sorted(LAYERS))
        ax.set_xticklabels([str(l) for l in sorted(LAYERS)], rotation=45 if len(LAYERS) > 10 else 0)

        # Grid and legend
        ax.grid(axis='both', alpha=0.3)
        if idx == 2:  # Only show legend on the last panel
            ax.legend(loc='best', framealpha=0.9, fontsize=9, ncol=2)

    plt.suptitle(f'Emotion Trajectories Across Layers: Case 4 Double-Diff with 95% Bootstrap CI',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    trajectory_path = OUTPUT_DIR / f'{QUESTION_MODULE}_emotion_trajectories_orthogonal.png'
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved orthogonal emotion trajectories to {trajectory_path}")

    plt.show()

else:
    # For linear probes, single plot
    trajectory_data = {emotion: {'layers': [], 'means': [], 'ci_lower': [], 'ci_upper': []}
                       for emotion in EMOTIONS}

    for layer in sorted(LAYERS):
        result = double_diff_results[layer]
        for emotion in EMOTIONS:
            trajectory_data[emotion]['layers'].append(layer)
            trajectory_data[emotion]['means'].append(result['mean_effect'][emotion])
            trajectory_data[emotion]['ci_lower'].append(result['bootstrap_ci'][emotion]['lower'])
            trajectory_data[emotion]['ci_upper'].append(result['bootstrap_ci'][emotion]['upper'])

    # Create trajectory plot
    fig, ax = plt.subplots(figsize=(12, 7))

    for emotion in EMOTIONS:
        layers = trajectory_data[emotion]['layers']
        means = trajectory_data[emotion]['means']
        ci_lower = trajectory_data[emotion]['ci_lower']
        ci_upper = trajectory_data[emotion]['ci_upper']

        color = EMOTION_COLORS[emotion]

        # Plot line with markers
        ax.plot(layers, means, marker='o', linewidth=2.5, markersize=8,
                color=color, label=emotion.capitalize(), alpha=0.9)

        # Add confidence interval band
        ax.fill_between(layers, ci_lower, ci_upper,
                         color=color, alpha=0.2, linewidth=0)

    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)

    # Formatting
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Double-Diff Effect', fontsize=13, fontweight='bold')
    ax.set_title(f'Emotion Trajectories Across Layers: {QUESTION_MODULE}\n' +
                 f'Case 4 Double-Diff with 95% Bootstrap CI',
                 fontsize=14, fontweight='bold')

    # Set x-axis to show only the layers we tested
    ax.set_xticks(sorted(LAYERS))
    ax.set_xticklabels([str(l) for l in sorted(LAYERS)])

    # Grid and legend
    ax.grid(axis='both', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=10, ncol=2)

    plt.tight_layout()

    # Save
    trajectory_path = OUTPUT_DIR / f'{QUESTION_MODULE}_emotion_trajectories.png'
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved emotion trajectories to {trajectory_path}")

    plt.show()

# %% Export Results to JSON
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

export_data = {
    'config': {
        'base_model': BASE_MODEL_NAME,
        'adapter_path': ADAPTER_PATH,
        'question_module': QUESTION_MODULE,
        'probe_type': PROBE_TYPE,
        'layers': LAYERS,
        'n_components': N_COMPONENTS,
        'activation_strategy': ACTIVATION_STRATEGY,
        'n_bootstrap': N_BOOTSTRAP,
        'ci_percentile': CI_PERCENTILE,
    },
    'prompts': {
        'n_dataset_prompts': len(dataset_prompts),
        'n_baseline_prompts': len(baseline_prompts),
        'dataset_examples': dataset_prompts[:3],
        'baseline_examples': baseline_prompts[:3],
    },
    'results_by_layer': {}
}

if PROBE_TYPE == "orthogonal":
    export_data['config']['orthogonal_representation'] = ORTHOGONAL_REPRESENTATION
    export_data['config']['orthogonality_weight'] = ORTHOGONALITY_WEIGHT

for layer in LAYERS:
    result = double_diff_results[layer]
    layer_export = {
        'averaged': {
            'mean_effect': result['mean_effect'],
            'bootstrap_ci': result['bootstrap_ci'],
            'emotion_ranking': [
                {'emotion': e, 'abs_value': float(v)}
                for e, v in result['emotion_ranking']
            ],
            'per_pair_effects': result['per_pair_effects']
        }
    }

    # Add separate user and assistant results if using orthogonal probes
    if PROBE_TYPE == "orthogonal":
        result_user = double_diff_results_user[layer]
        result_asst = double_diff_results_asst[layer]

        layer_export['user'] = {
            'mean_effect': result_user['mean_effect'],
            'bootstrap_ci': result_user['bootstrap_ci'],
            'emotion_ranking': [
                {'emotion': e, 'abs_value': float(v)}
                for e, v in result_user['emotion_ranking']
            ],
            'per_pair_effects': result_user['per_pair_effects']
        }

        layer_export['assistant'] = {
            'mean_effect': result_asst['mean_effect'],
            'bootstrap_ci': result_asst['bootstrap_ci'],
            'emotion_ranking': [
                {'emotion': e, 'abs_value': float(v)}
                for e, v in result_asst['emotion_ranking']
            ],
            'per_pair_effects': result_asst['per_pair_effects']
        }

    export_data['results_by_layer'][layer] = layer_export

# Save to JSON
json_path = OUTPUT_DIR / f'{QUESTION_MODULE}_double_diff_results.json'
with open(json_path, 'w') as f:
    json.dump(export_data, f, indent=2)

print(f"✓ Saved results to {json_path}")



# %%
