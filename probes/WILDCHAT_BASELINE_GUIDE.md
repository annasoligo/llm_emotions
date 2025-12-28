# WildChat Baseline Computation Guide

This guide explains how to compute probe baseline statistics from the WildChat conversational dataset.

## Overview

Baseline statistics help you interpret probe scores by showing what "typical" emotion scores look like on neutral, diverse conversational text. The process has two steps:

1. **Extract activations** from WildChat → Save to HDF5 (once per layer)
2. **Apply probes** to saved activations → Compute statistics (once per probe)

This separation makes it fast to compute baselines for new probes without re-extracting activations.

---

## Step 1: Extract WildChat Activations

**Run once per layer you want baselines for.**

```bash
# Extract activations for layer 31
python probes/scripts/compute_wildchat_baseline_activations.py \
    --layer 31 \
    --num-samples 512

# Or for multiple layers
for layer in 20 31 39 50 60; do
    python probes/scripts/compute_wildchat_baseline_activations.py --layer $layer
done
```

**What this does:**
- Loads 512 WildChat conversations from cached file
- Extracts activations at specified layer
- Saves multiple aggregations to HDF5:
  - `all_tokens`: Mean across all tokens (from position 20 onward)
  - `user_turn`: Mean across user turn tokens
  - `assistant_turn`: Mean across assistant turn tokens
  - `last_user_token`: Last token before assistant response
  - `first_assistant_token`: First token of assistant response
  - `between_turns`: Average of last_user and first_assistant

**Output:**
```
data/baselines/wildchat/google_gemma_3_27b_it/
├── layer20_activations.h5     # Saved activations [n_samples, hidden_dim]
├── layer20_stats.json          # Quick summary (mean/std)
├── layer31_activations.h5
├── layer31_stats.json
└── ...
```

**Options:**
- `--model`: Model name (default: `google/gemma-3-27b-it`)
- `--layer`: Layer to extract (required)
- `--num-samples`: How many conversations to use (default: all 512)
- `--wildchat-cache`: Path to cache file (default: believe-it-or-not repo location)
- `--output-dir`: Where to save (default: `data/baselines/wildchat`)

---

## Step 2: Compute Probe Baseline Statistics

**Run once per probe you want baselines for.**

```bash
# Compute baselines for layer 31, ortho weight 1000
python probes/scripts/compute_probe_baselines_from_wildchat.py \
    --layer 31 \
    --ortho-weight 1000.0

# For multiple probes
for weight in 1.0 10.0 100.0 1000.0; do
    python probes/scripts/compute_probe_baselines_from_wildchat.py \
        --layer 31 \
        --ortho-weight $weight
done
```

**What this does:**
- Loads saved activations from Step 1
- Loads orthogonal probe (user and assistant directions)
- Applies probes to all activation aggregations
- Computes statistics (mean, std, min, max, median) for each emotion

**Output:**
```
data/baselines/wildchat_probe_stats/google_gemma_3_27b_it/
├── layer31_raw_ortho1000.0_baseline_stats.json
├── layer31_raw_ortho100.0_baseline_stats.json
└── ...
```

**JSON structure:**
```json
{
  "model_name": "google/gemma-3-27b-it",
  "layer": 31,
  "num_samples": 512,
  "aggregations": {
    "all_tokens": {
      "user": {
        "happiness": {"mean": 0.123, "std": 0.456, ...},
        "anger": {"mean": -0.089, "std": 0.234, ...},
        ...
      },
      "assistant": {
        "happiness": {"mean": 0.234, "std": 0.345, ...},
        ...
      }
    },
    "user_turn": {...},
    "assistant_turn": {...},
    ...
  }
}
```

**Options:**
- `--model`: Model name (must match Step 1)
- `--layer`: Layer number (must match Step 1)
- `--ortho-weight`: Orthogonality weight (required)
- `--representation`: Probe type (default: `raw`)
- `--n-components`: For cPCA probes only
- `--activations-dir`: Where Step 1 saved activations
- `--probes-dir`: Where trained probes are located
- `--output-dir`: Where to save statistics

---

## Using Baselines in Analysis

### Quick Start: Load and Compare

```python
import json
import numpy as np

# Load baseline statistics
with open('data/baselines/wildchat_probe_stats/.../layer31_raw_ortho1000.0_baseline_stats.json') as f:
    baseline = json.load(f)

# Get baseline for a specific aggregation and emotion
baseline_mean = baseline['aggregations']['assistant_turn']['assistant']['happiness']['mean']
baseline_std = baseline['aggregations']['assistant_turn']['assistant']['happiness']['std']

# Compare your prompt's score to baseline
your_score = 2.5  # From token_level_analysis
z_score = (your_score - baseline_mean) / baseline_std

print(f"Your score: {your_score:.3f}")
print(f"Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"Z-score: {z_score:.2f} (deviation from neutral)")
```

### Integration with Token Analysis

Add this to your interactive notebook or script:

```python
# %% Load Baseline Statistics
BASELINE_FILE = Path(f"data/baselines/wildchat_probe_stats/google_gemma_3_27b_it/"
                     f"layer{LAYER}_raw_ortho{ORTHO_WEIGHT}_baseline_stats.json")

if BASELINE_FILE.exists():
    with open(BASELINE_FILE) as f:
        baseline_stats = json.load(f)
    print("✓ Baseline statistics loaded")
else:
    print(f"⚠ Baseline not found: {BASELINE_FILE}")
    print("  Run: python compute_wildchat_baseline_activations.py --layer", LAYER)
    print("       python compute_probe_baselines_from_wildchat.py --layer", LAYER, "--ortho-weight", ORTHO_WEIGHT)
    baseline_stats = None

# %% Compare to Baseline
if baseline_stats:
    print("\n" + "="*80)
    print("COMPARISON TO BASELINE")
    print("="*80)

    # Generated tokens comparison
    gen_positions = [pos for pos in sorted(assistant_scores.keys()) if pos > user_turn_end_pos]

    print("\nAssistant probe (generated tokens) vs WildChat baseline:")
    print(f"{'Emotion':12s} {'Your Score':>12s} {'Baseline':>12s} {'Diff':>10s} {'Z-score':>10s}")
    print("-" * 60)

    for emotion in EMOTIONS:
        your_score = np.mean([assistant_scores[pos][emotion] for pos in gen_positions])
        baseline_mean = baseline_stats['aggregations']['assistant_turn']['assistant'][emotion]['mean']
        baseline_std = baseline_stats['aggregations']['assistant_turn']['assistant'][emotion]['std']

        diff = your_score - baseline_mean
        z_score = diff / baseline_std if baseline_std > 0 else 0

        print(f"{emotion:12s} {your_score:>12.3f} {baseline_mean:>12.3f} {diff:>+10.3f} {z_score:>+10.2f}")
```

---

## Available Aggregation Types

Each aggregation gives you a different baseline perspective:

| Aggregation | Description | Use When |
|------------|-------------|----------|
| `all_tokens` | Mean across all tokens (from pos 20) | General baseline across full conversation |
| `user_turn` | Mean across user turn only | Comparing user input emotions |
| `assistant_turn` | Mean across assistant response | Comparing generated response emotions |
| `last_user_token` | Single token at end of user turn | Point-wise baseline before generation |
| `first_assistant_token` | First token of assistant response | Point-wise baseline at generation start |
| `between_turns` | Average of last_user + first_assistant | Baseline at turn boundary |

---

## Tips & Best Practices

### 1. **Choose the right aggregation**
- For analyzing **generated text**, use `assistant_turn` baseline
- For analyzing **user input**, use `user_turn` baseline
- For **overall conversation** comparison, use `all_tokens` baseline

### 2. **Interpret Z-scores**
- **|z| < 1**: Within normal range (close to neutral baseline)
- **|z| > 2**: Notably different from neutral conversations
- **|z| > 3**: Strong deviation (rare in neutral text)

### 3. **Re-use activations**
Once you've run Step 1 for a layer, you can compute baselines for ALL probes at that layer without re-extracting activations. This makes experimentation fast.

### 4. **Sample size**
- Default 512 samples gives stable statistics
- For faster iteration, use `--num-samples 100` in Step 1
- For production baselines, use all 512 samples

---

## Example Workflow

```bash
# 1. Extract activations for your layer (once)
python probes/scripts/compute_wildchat_baseline_activations.py --layer 31

# 2. Compute probe baselines (once per probe)
python probes/scripts/compute_probe_baselines_from_wildchat.py \
    --layer 31 \
    --ortho-weight 1000.0

# 3. Use in your analysis
# (Add baseline loading code to your notebook as shown above)
```

---

## Troubleshooting

### "Activations not found"
Run Step 1 first:
```bash
python probes/scripts/compute_wildchat_baseline_activations.py --layer <your_layer>
```

### "Probe not found"
Check available probes:
```bash
ls outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_*/
```

Train probe if needed:
```bash
python probes/scripts/training/train_orthogonal_conversation_probe.py \
    --layer <layer> \
    --representation raw \
    --ortho-weight <weight>
```

### "WildChat cache not found"
The cache file should be at:
```
/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens/wildchat_cache.jsonl
```

If missing, check the believe-it-or-not repository or specify a different path with `--wildchat-cache`.

---

## Next Steps

Once you have baseline statistics, you can:
1. Add baseline comparison to the interactive notebook
2. Create z-score visualizations
3. Flag anomalous emotional responses
4. Build a baseline library for multiple layers/models
