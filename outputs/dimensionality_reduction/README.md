# Dimensionality Reduction Results

This directory contains principal component analysis results from cPCA (Contrastive PCA) and Ratio PCA methods.

## Directory Structure

```
dimensionality_reduction/
├── cpca/                        # Contrastive PCA results
│   ├── tier_based/              # Text pair data (tiers: third_person, second_person, direct_address)
│   │   ├── google/              # Model vendor subdirectory
│   │   │   ├── gemma-3-27b-it_cpca.npz
│   │   │   └── gemma-3-27b-it_cpca_metadata.json
│   │   └── alpha_sweep/         # Different alpha parameter values
│   │       ├── alpha_1.0/
│   │       ├── alpha_5.0/       # Recommended (high alpha)
│   │       └── alpha_10.0/
│   │
│   ├── conversation_based/      # Conversation data
│   │   ├── global/              # Global conversation activations
│   │   └── regional/            # Regional splits
│   │       ├── user/            # User turn activations
│   │       ├── assistant/       # Assistant turn activations
│   │       ├── special_tokens_1/
│   │       └── special_tokens_2/
│   │
│   └── combined_ranges/         # Merged layer range results
│
├── ratio_pca/                   # Ratio PCA results
│   └── tier_based/
│       ├── k5/                  # k=5 components
│       ├── k10/                 # k=10 components
│       ├── k30/                 # k=30 components
│       └── k_tuning/            # Parameter sweep
│
└── comparisons/                 # Method comparison results
    ├── cpca_vs_ratio_pca_k5.json
    ├── cpca_vs_ratio_pca_k30.json
    └── method_comparison_plots/
```

## File Formats

### cPCA Results (.npz)

NumPy archive containing:

```python
{
    'components': np.ndarray,      # [n_layers, n_components, hidden_dim]
    'eigenvalues': np.ndarray,     # [n_layers, n_components]
    'explained_variance_ratio': np.ndarray,  # [n_layers, n_components]
    'alpha': float,                # Alpha parameter value
    'model_name': str,             # Model identifier
    'timestamp': str,              # Creation timestamp
}
```

### Metadata (.json)

Hyperparameters and settings:

```json
{
    "model_name": "google/gemma-3-27b-it",
    "data_type": "tier_based",
    "alpha": 5.0,
    "n_components": 64,
    "layers": [0, 1, 2, ..., 63],
    "timestamp": "2025-12-27T10:30:00",
    "script_version": "1.2.3"
}
```

## Usage

### Running cPCA

**Tier-based data (recommended settings):**

```bash
python -m probes.scripts.dimensionality_reduction.run_cpca \
    --data-path data/activations/texts_combined.h5 \
    --alpha 5.0 \
    --n-components 64 \
    --output-dir outputs/dimensionality_reduction/cpca/tier_based/alpha_sweep/alpha_5.0/google
```

**Conversation-based data (global):**

```bash
python -m probes.scripts.dimensionality_reduction.run_cpca \
    --data-path data/activations/conversations_combined.h5 \
    --alpha 5.0 \
    --output-dir outputs/dimensionality_reduction/cpca/conversation_based/global/google
```

**Regional conversation data:**

```bash
python -m probes.scripts.dimensionality_reduction.run_regional_cpca \
    --data-path data/activations/regional/user_turns.h5 \
    --alpha 5.0 \
    --region-name user \
    --output-dir outputs/dimensionality_reduction/cpca/conversation_based/regional/user/google
```

### Loading cPCA Components

```python
import numpy as np
from pathlib import Path

# Load cPCA results
cpca_path = Path("outputs/dimensionality_reduction/cpca/tier_based/google/gemma-3-27b-it_cpca.npz")
data = np.load(cpca_path)

components = data['components']     # [n_layers, n_components, hidden_dim]
eigenvalues = data['eigenvalues']   # [n_layers, n_components]

# Get components for a specific layer
layer_30_components = components[30]  # [n_components, hidden_dim]

print(f"Shape: {layer_30_components.shape}")
print(f"Top 10 eigenvalues at layer 30: {eigenvalues[30, :10]}")
```

### Using with output_config

```python
from probes.output_config import get_cpca_output_dir

# Get output directory for tier-based cPCA
output_dir = get_cpca_output_dir(
    data_type="tier_based",
    model_name="google/gemma-3-27b-it",
    alpha=5.0
)

print(f"cPCA results will be saved to: {output_dir}")
```

## Methods

### Contrastive PCA (cPCA)

Identifies directions that vary more in the foreground data (emotional text) than in the background data (neutral text).

**Key parameter: Alpha (α)**
- **α = 0**: Standard PCA
- **α = 1-2**: Mild contrastive effect
- **α = 5** (recommended): Strong contrastive effect, good for emotion detection
- **α = 10+**: Very strong, may overfit

**Advantages:**
- Focuses on emotion-relevant directions
- Better probe accuracy than raw activations
- Interpretable principal components

**Recommended configuration:**
- Alpha: 5.0
- Components: 64 (keep all, select top-k later)
- Layers: All layers (0-63 for Gemma-3-27B)

### Ratio PCA

Alternative dimensionality reduction focusing on variance ratios.

```bash
python -m probes.scripts.dimensionality_reduction.run_ratio_pca \
    --data-path data/activations/texts_combined.h5 \
    --k 10 \
    --output-dir outputs/dimensionality_reduction/ratio_pca/tier_based/k10
```

**K parameter tuning:**

```bash
python -m probes.scripts.dimensionality_reduction.tune_ratio_pca_k \
    --data-path data/activations/texts_combined.h5 \
    --k-values 3 5 10 20 30 50 \
    --output outputs/dimensionality_reduction/ratio_pca/tier_based/k_tuning/tuning_results.json
```

## Component Analysis

### Top Components by Eigenvalue

Higher eigenvalues indicate more variance explained:

```python
# Get top 10 components for layer 30
eigenvals = data['eigenvalues'][30]
top_10_indices = np.argsort(-eigenvals)[:10]

print("Top 10 components:")
for i, idx in enumerate(top_10_indices):
    print(f"  PC {idx}: eigenvalue = {eigenvals[idx]:.4f}")
```

### Explained Variance

```python
variance_ratios = data['explained_variance_ratio'][30]
cumulative_variance = np.cumsum(variance_ratios)

# Find how many components needed for 90% variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Components needed for 90% variance: {n_components_90}")
```

## Layer Analysis

cPCA components vary across layers:

**Early layers (0-20):**
- Capture surface-level linguistic features
- Lower eigenvalues
- Less emotion-specific

**Middle layers (30-40):**
- Peak emotion representation
- Highest eigenvalues
- Best probe accuracy
- **Recommended for emotion probes**

**Late layers (50-63):**
- Task-specific representations
- Good emotion separation but slightly noisier

## Regional cPCA (Conversations)

For conversation data, we compute cPCA separately for different regions:

### Global
All conversation tokens (user + assistant + special)

### Regional Splits
- **User turns**: Only user message tokens
- **Assistant turns**: Only assistant response tokens
- **Special tokens 1**: BOS/EOS/separator tokens
- **Special tokens 2**: Other special tokens

**Benefits:**
- Better disentanglement of user vs assistant emotions
- Cleaner orthogonal probe training
- More interpretable components

**Usage:**

```bash
# Run cPCA for all regions
python -m probes.scripts.dimensionality_reduction.run_regional_cpca \
    --base-data-path data/activations/conversations_combined.h5 \
    --regions user assistant special1 special2 \
    --alpha 5.0 \
    --output-base outputs/dimensionality_reduction/cpca/conversation_based/regional
```

## Method Comparison

Compare cPCA vs Ratio PCA:

```bash
python -m probes.scripts.evaluation.compare_pca_methods \
    --cpca-path outputs/dimensionality_reduction/cpca/tier_based/google/gemma-3-27b-it_cpca.npz \
    --ratio-pca-path outputs/dimensionality_reduction/ratio_pca/tier_based/k10/ \
    --output outputs/dimensionality_reduction/comparisons/cpca_vs_ratio_pca_k10.json
```

Results show cPCA typically achieves 2-3% higher probe accuracy.

## SLURM Scripts

For large models, run cPCA on GPU cluster:

```bash
# Tier-based cPCA (high alpha)
sbatch probes/scripts/slurm_jobs/dimensionality_reduction/run_cpca_high_alpha.sh

# Conversation cPCA (global, parallel across layers)
sbatch probes/scripts/slurm_jobs/dimensionality_reduction/run_conversation_cpca_global_parallel.sh

# All conversation cPCA (global + regional pipeline)
sbatch probes/scripts/slurm_jobs/dimensionality_reduction/run_all_conversation_cpca.sh
```

## Storage

- **cPCA files (.npz)**: 10-100 MB per model
- **Keep all layer results** for flexibility
- **Archive old alpha sweeps** after identifying best value
- **Document hyperparameters** in metadata.json

## Related Documentation

- [Probe Training](../probes/README.md) - Using cPCA results for probes
- [Interpretations](../interpretations/README.md) - Understanding PC meanings
- [Comparison Results](./comparisons/) - Method comparison analyses

---

Last updated: 2025-12-27
