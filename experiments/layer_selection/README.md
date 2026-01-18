# Layer Selection for Emotion Steering Vectors

This module identifies the most significant layers for emotion steering in Gemma 3 27B by computing 3 metrics across all 62 layers.

## Overview

When performing activation steering, choosing the right layer is critical. Different layers may have different effects on model behavior. This module helps identify optimal layers by computing:

1. **Attribution Patching (KL Divergence)**: Measures how much each layer contributes to the steering effect via gradient-based attribution.

2. **Variance Ratio**: Compares variance along steering directions to random directions. Ratio > 1 indicates the steering direction captures meaningful signal at that layer.

3. **PCA Alignment**: Measures how much the steering vector aligns with the top principal components of the activation space. Higher alignment suggests the direction is well-represented.

## Files

```
experiments/layer_selection/
├── __init__.py                   # Module docstring
├── compute_layer_metrics.py      # Main computation script
├── eval_prompts.py               # 100 neutral evaluation prompts
├── plot_layer_metrics.py         # Visualization script
├── slurm_compute_metrics.sh      # SLURM job for GPU computation
├── README.md                     # This file
└── results/                      # Output directory
    ├── layer_metrics.npz
    ├── attribution_scores_by_layer.png
    ├── variance_ratio_by_layer.png
    ├── pca_alignment_by_layer.png
    └── summary.txt
```

## Usage

### Run on SLURM cluster

```bash
sbatch experiments/layer_selection/slurm_compute_metrics.sh
```

### Run locally (requires GPU)

```bash
python -m experiments.layer_selection.compute_layer_metrics \
    --model google/gemma-3-27b-it \
    --h5-path outputs/data/activations/texts_combined.h5 \
    --output-dir experiments/layer_selection/results/
```

### Skip model loading (faster, no attribution)

```bash
python -m experiments.layer_selection.compute_layer_metrics \
    --h5-path outputs/data/activations/texts_combined.h5 \
    --output-dir experiments/layer_selection/results/ \
    --skip-attribution
```

### Generate plots only

```bash
python -m experiments.layer_selection.plot_layer_metrics \
    --results experiments/layer_selection/results/layer_metrics.npz \
    --output-dir experiments/layer_selection/results/
```

## Data Flow

```
texts_combined.h5 (all 62 layers, 6 emotions)
        │
        ▼
load_or_compute_steering_vectors()
        │
        ▼
{layer: {emotion: unit_vector[5376]}}
        │
    ┌───┴───┐
    ▼       ▼
  Model   Cached Activations
    │       │
    ▼       ▼
Attribution  Variance Ratio + PCA Alignment
    │           │
    └─────┬─────┘
          ▼
    layer_metrics.npz
          │
          ▼
      3 PNG plots
```

## Output Format

The results are saved in `layer_metrics.npz` with the following keys:
- `attribution_{emotion}`: [n_layers] attribution scores for each emotion
- `variance_ratio_{emotion}`: [n_layers] variance ratios for each emotion
- `pca_alignment_{emotion}`: [n_layers] PCA alignment values for each emotion
- `attribution_average`: [n_layers] average attribution across emotions
- `variance_ratio_average`: [n_layers] average variance ratio
- `pca_alignment_average`: [n_layers] average PCA alignment
- `emotions`: list of emotion names
- `n_layers`: number of layers

## Metrics Interpretation

### Attribution Scores
- **Higher is better**: Layers with high attribution scores contribute more to the steering effect
- Useful for identifying which layers are most influential

### Variance Ratio
- **> 1.0 is meaningful**: The steering direction captures more variance than random
- **>> 1.0 is optimal**: Strong signal that this layer encodes emotion-relevant information

### PCA Alignment
- **0-1 scale**: Fraction of steering vector in top-k PC subspace
- **Higher is better**: Well-aligned vectors are more likely to produce coherent steering
- **Lower can be problematic**: May steer in directions the model doesn't naturally explore

## Evaluation Prompts

The 100 neutral prompts are designed to be emotionally ambiguous:
- 20 work/professional scenarios
- 20 daily life situations
- 20 environmental observations
- 20 social interactions
- 20 ambiguous events

These can be steered toward any emotion, making them ideal for testing.

## Expected Results

For Gemma 3 27B, we typically expect:
- Middle layers (15-40) to show highest variance ratios
- Later layers may show higher PCA alignment
- Attribution patterns depend on target layer choice

## Dependencies

- `numpy`
- `torch`
- `transformers`
- `h5py`
- `scikit-learn`
- `matplotlib`
- `tqdm`
