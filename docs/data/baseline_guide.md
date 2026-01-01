# Baseline Statistics Guide

This document explains the two sets of baseline statistics used for probe score normalization.

---

## Overview

The codebase uses **pre-computed baseline statistics** to z-score normalize probe outputs. There are **two different baseline datasets**:

1. **WildChat Baselines** (original, legacy)
2. **Alpaca Baselines** (current, actively used)

---

## 1. WildChat Baselines

### Location
```
/workspace-vast/annas/git/research-tools/data/baselines/wildchat/google_gemma_3_27b_it/
```

### Structure
```
wildchat/
└── google_gemma_3_27b_it/
    ├── layer0_activations.h5
    ├── layer1_activations.h5
    ├── ...
    └── layer63_activations.h5
```

Each `.h5` file contains:
- Raw activations from WildChat dataset
- Multiple aggregation types:
  - `all_tokens`: All tokens in conversation
  - `user_turn`: Only user turns
  - `assistant_turn`: Only assistant turns
  - `last_user_token`: Last token of user turns
  - `first_assistant_token`: First token of assistant turns
  - `between_turns`: Between user and assistant turns

### Generation Scripts
- `probes/scripts/compute_wildchat_baseline_activations.py` - Extracts activations
- `probes/scripts/compute_probe_baselines_from_wildchat.py` - Computes probe scores
- `probes/scripts/check_wildchat_progress.sh` - Monitors progress

### Current Usage
**⚠️ NOT ACTIVELY USED** in current experiments!

The `WildChatBaselineLoader` class defaults to this path, but **all actual usage overrides it** to use Alpaca baselines instead.

**Where it's referenced:**
- `probes/scripts/wildchat_baseline_loader.py:20` - Default parameter (overridden)
- `probes/scripts/TOKEN_LEVEL_README.md` - Documentation (outdated)
- `probes/scripts/MODEL_DIFF_README.md` - Documentation (outdated)

---

## 2. Alpaca Baselines

### Location (Current Version)
```
/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it/
```

### Location (Legacy Version)
```
/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b/
```

### Structure
```
alpaca_gemma27b_v2/
└── google_gemma_3_27b_it/
    ├── layer0_activations.h5
    ├── layer1_activations.h5
    ├── ...
    └── layer63_activations.h5
```

Similar structure to WildChat, but computed from **Alpaca dataset** instead.

### Why Alpaca Instead of WildChat?

**Reason:** Alpaca provides a more neutral, instruction-following baseline.

- **WildChat**: Real user conversations (more varied, emotional, sometimes toxic)
- **Alpaca**: Clean instruction-following dataset (neutral, task-focused)

Using Alpaca as baseline means:
- Probe scores measure **deviation from neutral instruction-following**
- Better for detecting emotional content in target prompts
- More stable baseline statistics

### Generation Scripts
- `probes/scripts/generate_alpaca_responses.py` - Generate Alpaca responses
- `probes/scripts/compute_baseline_activations_from_jsonl.py` - Extract activations
- `probes/scripts/create_alpaca_baseline.sh` - End-to-end script
- `probes/scripts/slurm_alpaca_baseline.sh` - SLURM batch job
- `probes/scripts/slurm_alpaca_from_wildchat.sh` - V2 generation script

### Current Usage
**✅ ACTIVELY USED** in all current experiments!

**Where it's used:**
1. **Model Diffing Analysis**
   - `probes/scripts/model_diff_analysis_v2.py:108`
   - Path: `alpaca_gemma27b_v2/google_gemma_3_27b_it`

2. **Token-Level Analysis**
   - `probes/scripts/token_level_experiment_v2.py:112`
   - Path: `alpaca_gemma27b_v2/google_gemma_3_27b_it`

3. **Emotion Onset Probes**
   - `elicitation/scripts/run_emotion_onset_probes.py:72`
   - Path: `alpaca_gemma27b_v2/google_gemma_3_27b_it`

4. **Dashboard**
   - `eval_dashboard/probe_configs.py:104`
   - Path: `alpaca_gemma27b_v2/google_gemma_3_27b_it`

---

## How Baseline Normalization Works

### Step 1: Extract Activations from Baseline Dataset
```python
# Run model on Alpaca dataset
# Extract activations at each layer
# Save to layer{N}_activations.h5
```

### Step 2: Compute Statistics
```python
# For each layer:
mean = np.mean(activations, axis=0)  # [hidden_dim]
std = np.std(activations, axis=0)    # [hidden_dim]
```

### Step 3: Normalize Probe Scores
```python
# When applying probes:
raw_score = probe(activation)  # Raw probe output
normalized_score = (raw_score - baseline_mean) / baseline_std  # Z-score
```

### Result
- Probe scores in **standard deviation (σ) units**
- `score = 0σ` means "same as baseline"
- `score = +2σ` means "2 std devs above baseline"
- `score = -2σ` means "2 std devs below baseline"

---

## Class: WildChatBaselineLoader

Despite the name, this class is used for **both WildChat and Alpaca baselines**.

**Name is misleading!** It should be called `BaselineLoader` but was originally designed for WildChat.

### Usage
```python
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader

# Create loader (works with any baseline dir)
baseline_loader = WildChatBaselineLoader(
    baseline_dir=Path("data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it"),
    aggregation_type="all_tokens"
)

# Option 1: Normalize activations (rarely used now)
normalized_acts = baseline_loader.normalize_activations(activations, layer=30)

# Option 2: Normalize probe scores (commonly used)
baseline_stats = baseline_loader.compute_probe_score_baselines(
    probe_inference=probe_inference,
    layers=[20, 21, 22, ...],
    probe_type="orthogonal",
    return_std=True
)

normalized_scores = (raw_scores - baseline_stats['mean'][-1]) / baseline_stats['std'][-1]
```

---

## Key Methods

### 1. `load_layer_stats(layer)`
Load activation statistics for a layer.

**Returns:**
```python
{
    'mean': np.ndarray,  # [hidden_dim]
    'std': np.ndarray    # [hidden_dim]
}
```

### 2. `normalize_activations(activations, layer)`
Z-score normalize activations using baseline stats.

**Note:** This was part of "Stage 1 normalization" which we removed in the refactoring.

### 3. `compute_probe_score_baselines(probe_inference, layers, ...)`
Compute probe score statistics by applying probes to baseline activations.

**Returns:**
```python
{
    'mean': {
        30: np.ndarray,  # [n_emotions]
        31: np.ndarray,
        ...,
        -1: np.ndarray   # Layer-averaged (used for normalization)
    },
    'std': {
        30: np.ndarray,
        31: np.ndarray,
        ...,
        -1: np.ndarray   # Layer-averaged
    }
}
```

**This is what we use** for probe score normalization.

---

## Comparison: WildChat vs Alpaca

| Aspect | WildChat | Alpaca |
|--------|----------|--------|
| **Source** | Real user conversations | Instruction-following dataset |
| **Emotion Level** | Varied (some emotional/toxic) | Neutral (task-focused) |
| **Use Case** | Original baseline | Current baseline (better) |
| **Status** | Legacy (not used) | Active (all experiments) |
| **Dataset Size** | Large | Smaller but cleaner |
| **Baseline Quality** | Noisy | Stable |

---

## Migration History

### Phase 1: Original (WildChat)
- Used `data/baselines/wildchat/`
- Baseline from real conversations
- More noise in baseline statistics

### Phase 2: Alpaca V1
- Switched to `data/baselines/alpaca_gemma27b/`
- Cleaner baseline from instruction-following

### Phase 3: Alpaca V2 (Current)
- Updated to `data/baselines/alpaca_gemma27b_v2/`
- Improved generation pipeline
- Currently used in all experiments

---

## Which Baseline Should You Use?

**✅ Use Alpaca V2** (current standard)
```python
BASELINE_DIR = Path("/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it")
```

**Reasons:**
- More neutral baseline (better for emotion detection)
- Stable statistics
- Used by all current experiments
- Better for measuring deviations from "normal" instruction-following

---

## Regenerating Baselines

### For Alpaca Baselines (Recommended)
```bash
# Full pipeline
bash probes/scripts/create_alpaca_baseline.sh

# Or step-by-step:
# 1. Generate Alpaca responses
python probes/scripts/generate_alpaca_responses.py

# 2. Extract activations
python probes/scripts/compute_baseline_activations_from_jsonl.py \\
    --input data/alpaca_responses.jsonl \\
    --output data/baselines/alpaca_gemma27b_v2/
```

### For WildChat Baselines (Legacy)
```bash
# Full pipeline (legacy)
python probes/scripts/compute_wildchat_baseline_activations.py \\
    --output-dir data/baselines/wildchat
```

---

## TODO: Cleanup Tasks

### 1. Rename WildChatBaselineLoader
**Current name is confusing!**

Should be:
- `BaselineLoader` or `ProbeBaselineLoader`
- Works with any baseline directory (not just WildChat)

### 2. Update Documentation
Files referencing WildChat baselines:
- `probes/scripts/TOKEN_LEVEL_README.md:593` - Update to Alpaca
- `probes/scripts/MODEL_DIFF_README.md:171` - Update to Alpaca

### 3. Consider Removing WildChat Baselines
If not actively used, consider:
- Archiving the WildChat baseline directory
- Removing unused generation scripts
- Or document as "legacy" option

---

## Summary

**Current Standard:**
- **Baseline Dataset**: Alpaca V2
- **Location**: `data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it/`
- **Used By**: All experiments (model_diff, token_level, dashboard, emotion_onset)
- **Loader Class**: `WildChatBaselineLoader` (despite the name!)
- **Purpose**: Provide neutral baseline for z-score normalization of probe scores

**Legacy (Not Used):**
- **Baseline Dataset**: WildChat
- **Location**: `data/baselines/wildchat/google_gemma_3_27b_it/`
- **Status**: Referenced in code defaults but overridden everywhere
