# Orthogonality-Regularized Probes

This directory contains an implementation of linear emotion probes with orthogonality regularization to neutral principal components.

## Core Idea

The key innovation is adding a penalty term to the probe training loss that discourages probe weights from aligning with neutral-dominated principal components:

```
L_total = L_emotion + λ * ||U^T @ w||²
```

Where:
- `L_emotion`: Standard cross-entropy loss for emotion classification
- `U`: Matrix of neutral PCs `[hidden_dim, k]`
- `w`: Probe weights `[n_classes, hidden_dim]`
- `λ`: Orthogonality penalty weight (hyperparameter)

This forces the probe to learn emotion representations that are orthogonal to neutral content structure.

## Files Created

1. **`orthogonal_regularized_probe.py`**
   - `OrthogonalRegularizedProbe`: PyTorch module with orthogonality penalty
   - `train_orthogonal_regularized_probe()`: Training function with early stopping
   - Computes detailed orthogonality metrics for analysis

2. **`scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py`**
   - Computes neutral PCs using ratio-based PCA method
   - Identifies top-k PCs with highest `neutral_var / diff_var` ratio
   - Saves neutral PCs for use in probe training

3. **`scripts/training/train_orthogonal_regularized_probe.py`**
   - End-to-end training script for text-based probes
   - Loads activations from `texts.h5`
   - Loads pre-computed neutral PCs
   - Trains probe with configurable λ
   - Saves results and detailed summaries

4. **`scripts/training/test_orthogonal_regularized_probe.py`**
   - Test script with synthetic data
   - Validates the full pipeline
   - Useful for debugging and development

## Usage

### Step 1: Compute Neutral PCs

First, identify the neutral-dominated principal components for your layers of interest:

```bash
# Single layer
python probes/scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py \
  --data data/activations/texts.h5 \
  --layer 20 \
  --k 20 \
  --output probes/results/neutral_pcs/

# Multiple layers
python probes/scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py \
  --data data/activations/texts.h5 \
  --layers 10 15 20 25 30 \
  --k 20 \
  --output probes/results/neutral_pcs/

# All layers (0-31)
python probes/scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py \
  --data data/activations/texts.h5 \
  --layer-range 0 32 \
  --k 20 \
  --output probes/results/neutral_pcs/
```

**Parameters:**
- `--k`: Number of neutral PCs to extract (default: 20)
- `--n-pcs-all`: Number of PCs for ratio analysis (default: 100)
- `--tiers`: Specific tiers to include (optional)
- `--max-samples`: Limit samples for faster computation (optional)

**Output:**
- `probes/results/neutral_pcs/layer_<N>_neutral_pcs_k<K>.npy`: Neutral PCs for each layer
- `probes/results/neutral_pcs/config_k<K>.json`: Configuration and diagnostic info

### Step 2: Train Orthogonality-Regularized Probe

Train a probe with orthogonality regularization:

```bash
# With orthogonality regularization
python probes/scripts/training/train_orthogonal_regularized_probe.py \
  --data data/activations/texts.h5 \
  --layer 20 \
  --neutral-pcs probes/results/neutral_pcs/layer_20_neutral_pcs_k20.npy \
  --lambda-ortho 1.0 \
  --output-dir probes/results/orthogonal_regularized_probes/

# Baseline (no regularization)
python probes/scripts/training/train_orthogonal_regularized_probe.py \
  --data data/activations/texts.h5 \
  --layer 20 \
  --lambda-ortho 0.0 \
  --output-dir probes/results/orthogonal_regularized_probes/baseline/
```

**Key Parameters:**
- `--lambda-ortho`: Orthogonality penalty weight (try: 0.01, 0.1, 1.0, 10.0, 100.0)
- `--neutral-pcs`: Path to neutral PCs file (if not provided, trains standard probe)
- `--max-epochs`: Maximum training epochs (default: 500)
- `--patience`: Early stopping patience (default: 10)
- `--weight-decay`: L2 regularization (default: 1.0)

**Output:**
- `.pkl` file with trained model, predictions, and metrics
- `_summary.txt` file with readable summary

### Step 3: Hyperparameter Tuning

Sweep over λ values to find the best trade-off:

```bash
# Lambda sweep
for lambda in 0.01 0.1 1.0 10.0 100.0; do
    python probes/scripts/training/train_orthogonal_regularized_probe.py \
        --data data/activations/texts.h5 \
        --layer 20 \
        --neutral-pcs probes/results/neutral_pcs/layer_20_neutral_pcs_k20.npy \
        --lambda-ortho $lambda \
        --output-dir probes/results/orthogonal_regularized_probes/lambda_sweep/
done
```

Sweep over k (number of neutral PCs):

```bash
# First compute PCs for different k values
for k in 5 10 20 50 100; do
    python probes/scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py \
        --data data/activations/texts.h5 \
        --layer 20 \
        --k $k \
        --output probes/results/neutral_pcs/
done

# Then train with each
for k in 5 10 20 50 100; do
    python probes/scripts/training/train_orthogonal_regularized_probe.py \
        --layer 20 \
        --neutral-pcs probes/results/neutral_pcs/layer_20_neutral_pcs_k${k}.npy \
        --lambda-ortho 1.0 \
        --output-dir probes/results/orthogonal_regularized_probes/k_sweep/
done
```

## Using via SLURM

Create a SLURM script for batch processing:

```bash
#!/bin/bash
#SBATCH --job-name=ortho_probe
#SBATCH --output=logs/ortho_probe_%j.out
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Step 1: Compute neutral PCs (do this once)
python probes/scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py \
  --data data/activations/texts.h5 \
  --layer-range 0 32 \
  --k 20 \
  --output probes/results/neutral_pcs/

# Step 2: Train probes with different lambda values
for lambda in 0.01 0.1 1.0 10.0 100.0; do
    for layer in 10 15 20 25 30; do
        python probes/scripts/training/train_orthogonal_regularized_probe.py \
            --data data/activations/texts.h5 \
            --layer $layer \
            --neutral-pcs probes/results/neutral_pcs/layer_${layer}_neutral_pcs_k20.npy \
            --lambda-ortho $lambda \
            --output-dir probes/results/orthogonal_regularized_probes/sweep/
    done
done
```

## Evaluating Results

### Key Metrics

The training script reports several orthogonality metrics:

1. **Frobenius Norm**: `||U^T @ w||_F`
   - Raw magnitude of projection onto neutral subspace
   - Lower is better

2. **Normalized Overlap**: `||U^T @ w|| / ||w||`
   - Ranges from 0 (fully orthogonal) to 1 (fully aligned)
   - Lower is better
   - Most interpretable metric

3. **Neutral Fraction**: `||U @ U^T @ w|| / ||w||`
   - Fraction of probe weight lying in neutral subspace
   - Lower is better

### Analysis Script Template

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load results
baseline = pickle.load(open("results/.../baseline.pkl", "rb"))
ortho_results = {
    lambda_val: pickle.load(open(f"results/.../lambda{lambda_val}.pkl", "rb"))
    for lambda_val in [0.01, 0.1, 1.0, 10.0, 100.0]
}

# Compare accuracies
print("Lambda | Test Acc | Ortho Norm | Neutral Frac")
print("-" * 50)
print(f"Baseline | {baseline['test_accuracy']:.4f} | N/A | N/A")
for lambda_val, result in ortho_results.items():
    print(
        f"{lambda_val:6.2f} | "
        f"{result['test_accuracy']:.4f} | "
        f"{result['ortho_metrics']['ortho_normalized']:.4f} | "
        f"{result['ortho_metrics']['neutral_fraction']:.4f}"
    )

# Plot Pareto frontier
lambdas = list(ortho_results.keys())
accs = [ortho_results[l]['test_accuracy'] for l in lambdas]
orthos = [ortho_results[l]['ortho_metrics']['ortho_normalized'] for l in lambdas]

plt.figure(figsize=(8, 6))
plt.scatter(orthos, accs)
for i, l in enumerate(lambdas):
    plt.annotate(f"λ={l}", (orthos[i], accs[i]))
plt.xlabel("Orthogonality Violation (normalized)")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Orthogonality Trade-off")
plt.grid(True)
plt.savefig("pareto_frontier.png")
```

## Expected Behavior

Based on the design:

1. **Baseline (λ=0)**:
   - Highest accuracy
   - High neutral fraction (probe uses neutral PCs freely)

2. **Low λ (0.01-0.1)**:
   - Accuracy similar to baseline
   - Moderate reduction in neutral fraction

3. **Medium λ (1.0-10.0)**:
   - Small accuracy drop (1-3%)
   - Significant reduction in neutral fraction (50%+)
   - **Sweet spot** likely here

4. **High λ (100.0+)**:
   - Larger accuracy drop
   - Very low neutral fraction (strong orthogonality)
   - May be over-regularized

## Integration with Existing Code

The new probe type is fully compatible with existing infrastructure:

```python
from probes.methods import train_orthogonal_regularized_probe

# Load neutral PCs
neutral_pcs = np.load("probes/results/neutral_pcs/layer_20_neutral_pcs_k20.npy")

# Train probe
results = train_orthogonal_regularized_probe(
    train_activations=train_acts,
    train_labels=train_labels,
    test_activations=test_acts,
    test_labels=test_labels,
    neutral_pcs=neutral_pcs,
    lambda_ortho=1.0,
    device="cuda",
)

# Access results
model = results['model']
test_accuracy = results['test_accuracy']
ortho_metrics = results['ortho_metrics']
```

## Future Extensions

Possible improvements to explore:

1. **Per-emotion λ**: Different regularization strength for different emotions
2. **Layer-specific k**: Adapt number of neutral PCs per layer
3. **Dynamic λ scheduling**: Start high, decay during training
4. **Soft vs hard orthogonality**: Current is soft (penalty), could do hard (projection)
5. **Multi-task**: Combine with existing user/assistant orthogonality constraints

## Troubleshooting

**Issue**: Accuracy drops too much with regularization

**Solutions**:
- Reduce λ
- Reduce k (use fewer neutral PCs)
- Increase weight_decay (more L2 regularization)
- Check if neutral PCs actually capture neutral content (inspect ratios)

**Issue**: Orthogonality metrics not improving

**Solutions**:
- Increase λ
- Check that neutral_pcs are being loaded correctly
- Verify neutral_pcs.shape[0] matches activation dimensionality

**Issue**: Training is unstable

**Solutions**:
- Reduce learning rate
- Reduce λ
- Check for NaN in neutral_pcs
- Ensure neutral_pcs are properly normalized

## References

This implementation is inspired by:
- Domain adversarial training (Ganin et al., 2016)
- Contrastive PCA (Abid et al., 2018)
- Your existing ratio-based PCA method (pca_ratio.py)

The key difference from these approaches is simplicity: instead of adversarial training or contrastive covariance, we directly penalize alignment with pre-computed neutral directions.
