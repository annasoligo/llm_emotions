# WildChat Baseline - Quick Start

## What's Running

**SLURM Job 93043** is extracting WildChat baseline activations for layers 20, 31, 39, 50, 60 in a **SINGLE forward pass** (efficient!).

This extracts 512 conversational samples with 6 aggregation types per layer:
- `all_tokens` - Mean from position 20 onward
- `user_turn` - Mean across user tokens (from pos 20)
- `assistant_turn` - Mean across assistant tokens
- `last_user_token` - Last token before assistant
- `first_assistant_token` - First generated token
- `between_turns` - Average of last two

## Check Progress

```bash
./probes/scripts/check_wildchat_progress.sh
```

Or:
```bash
squeue -u annas | grep wildchat
tail -f probes/logs/wildchat_baseline_93043.log
```

## Once Complete

### Step 1: Compute Probe Baselines

Apply your trained probes to the saved activations:

```bash
python probes/scripts/compute_probe_baselines_from_wildchat.py \
    --layer 31 \
    --ortho-weight 1000.0
```

This loads the saved activations (fast!) and computes emotion score statistics.

### Step 2: Use in Analysis

Load baseline stats in your notebook:

```python
import json

# Load baseline
with open('data/baselines/wildchat_probe_stats/google_gemma_3_27b_it/'
          'layer31_raw_ortho1000.0_baseline_stats.json') as f:
    baseline = json.load(f)

# Get stats for specific aggregation/emotion
mean = baseline['aggregations']['assistant_turn']['assistant']['happiness']['mean']
std = baseline['aggregations']['assistant_turn']['assistant']['happiness']['std']

# Compute z-score
your_score = 2.5
z_score = (your_score - mean) / std
print(f"Z-score: {z_score:.2f} std from neutral baseline")
```

## Files Created

After job completes:
```
data/baselines/wildchat/google_gemma_3_27b_it/
├── layer20_activations.h5     # [512, 5632] arrays for each aggregation
├── layer20_stats.json          # Quick summary (mean/std)
├── layer31_activations.h5
├── layer31_stats.json
├── layer39_activations.h5
├── layer39_stats.json
├── layer50_activations.h5
├── layer50_stats.json
├── layer60_activations.h5
└── layer60_stats.json
```

## Adding More Layers

If you need baselines for additional layers:

```bash
# Extract additional layers (reuses model loading)
sbatch probes/scripts/slurm_wildchat_baselines.sh "0,10,25,35,45,55,61"

# Or ALL layers (62 total, ~30min)
sbatch probes/scripts/slurm_wildchat_baselines.sh "all"
```

## Key Advantage

**One extraction, many probes**: Once activations are saved, you can quickly compute baselines for ANY probe without rerunning the model. This makes probe iteration fast!
