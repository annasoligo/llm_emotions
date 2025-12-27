# Ratio-Based PCA (PCA v3)

Alternative to contrastive PCA (cPCA) for neutral signal removal.

## Overview

**Ratio PCA** removes neutral-dominated principal components based on the ratio of neutral variance to difference variance. Unlike cPCA which requires alpha tuning, ratio PCA uses a simpler approach:

1. Run PCA on combined emotional + neutral activations
2. For each PC, compute `ratio = var(neutral_proj) / var(diff_proj)`
3. Remove the top-k PCs with highest ratio (neutral-dominated directions)
4. Project out these directions from the difference vectors

## Key Differences from cPCA

| Aspect | cPCA | Ratio PCA |
|--------|------|-----------|
| **Method** | Alpha-tuned covariance: `C_emo - α*C_neu` | Remove high-ratio PCs |
| **Hyperparameter** | Alpha (requires grid search per layer) | k (PCs to remove, typically 10) |
| **Computation** | Eigendecomposition of contrastive covariance | Standard PCA + ratio filtering |
| **Interpretability** | Alpha values hard to interpret | Ratio shows neutral dominance |
| **Speed** | Slower (alpha tuning per layer) | Faster (single PCA) |

## Usage

### Run Ratio PCA

```bash
# Basic usage
python -m probes.scripts.run_ratio_pca \
    --activations data/activations/texts_combined.h5 \
    --output probes/results/ratio_pca_tier_data/ \
    --k 10 \
    --n-components 50

# SLURM job
sbatch probes/scripts/slurm_jobs/run_ratio_pca_tier_data.sh
```

### Compare with cPCA

```bash
python -m probes.scripts.compare_pca_methods \
    --ratio-pca probes/results/ratio_pca_tier_data/model_ratio_pca.npz \
    --cpca probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz \
    --ratio-summary probes/results/ratio_pca_tier_data/model_summary.json \
    --cpca-summary probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_summary.json \
    --output probes/results/ratio_vs_cpca_comparison/ \
    --heatmap-layer 30
```

### Use Results in Downstream Tasks

The ratio PCA output format is compatible with cPCA, so you can use it directly in:
- Auto-interpretation scripts
- Probe training
- Emotion steering

```bash
# Example: Run autointerp on ratio PCA components
python -m probes.scripts.autointerp_pcs \
    --cpca probes/results/ratio_pca_tier_data/model_ratio_pca.npz \
    --activations data/activations/texts_combined.h5 \
    --texts data/texts_combined_pairs.jsonl \
    --output probes/results/autointerp_ratio_pca/ \
    --layers 30 \
    --pcs 0 1 2 3 4
```

## Algorithm Details

### Ratio Computation

For each principal component PC_i:

```
neutral_proj = neutral @ PC_i     # Project neutral activations
diff_proj = diffs @ PC_i           # Project differences

neutral_var = var(neutral_proj)    # Variance in neutral
diff_var = var(diff_proj)          # Variance in differences

ratio_i = neutral_var / diff_var   # High = neutral-dominated
```

### Neutral Removal

```python
# 1. Find top-k highest ratio PCs
top_k_indices = argsort(ratio)[-k:]
pcs_to_remove = all_pcs[top_k_indices]

# 2. Project out neutral-dominated directions
projections = diffs @ pcs_to_remove.T
cleaned_diffs = diffs - projections @ pcs_to_remove

# 3. Run final PCA on cleaned differences
pca.fit(cleaned_diffs)
final_components = pca.components_
```

## Diagnostic Metrics

The summary JSON includes:

```json
{
  "diagnostics": {
    "avg_ratio_removed": 1.187,    // Average ratio of removed PCs
    "avg_ratio_kept": 0.117,       // Average ratio of kept PCs
    "ratio_improvement": 10.12     // Improvement factor
  },
  "removal_info_per_layer": {
    "30": {
      "mean_ratio_removed": 1.245,
      "mean_ratio_kept": 0.118,
      "top_k_ratios": [1.5, 1.4, ...]  // Ratios of removed PCs
    }
  }
}
```

## When to Use Each Method

**Use cPCA when:**
- You want theoretically grounded contrastive learning
- You need fine-grained control (alpha per layer)
- You're comparing with published cPCA work

**Use Ratio PCA when:**
- You want a simpler, more interpretable approach
- You want faster computation (no alpha tuning)
- You want explicit diagnostics on what's being removed

## Implementation Files

- `probes/methods/pca_ratio.py` - Core implementation
- `probes/scripts/run_ratio_pca.py` - Execution script
- `probes/scripts/compare_pca_methods.py` - Comparison tool
- `probes/scripts/slurm_jobs/run_ratio_pca_tier_data.sh` - SLURM job

## Origin

Adapted from PCA v3 implementation in `/workspace-vast/annas/git/believe-it-or-not/pc_probes/pca_v3.py`.
