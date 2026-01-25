# Orthogonal Regularization Run Summary

**Date:** January 1, 2026
**Status:** ✅ Jobs submitted and running

---

## Jobs Submitted

### 1. Test Job (ID: 99299)
**Status:** Running on node-2
**Script:** `probes/scripts/slurm_jobs/training/test_orthogonal_regularized_single.sh`
**Purpose:** Quick validation on layer 20 with λ=1.0
**Outputs:**
- Neutral PCs: `probes/results/neutral_pcs_test/`
- Baseline probe: `probes/results/orthogonal_regularized_probes_test/baseline/`
- Orthogonal probe: `probes/results/orthogonal_regularized_probes_test/lambda_1.0/`
- Log: `probes/logs/ortho_test_99299.log`

### 2. Neutral PCs Computation (ID: 99300)
**Status:** Running on node-12
**Script:** `probes/scripts/slurm_jobs/dimensionality_reduction/compute_neutral_pcs.sh`
**Configuration:**
- Layers: 10, 15, 20, 25, 30
- k = 20 (number of neutral PCs)
- n_pcs_all = 100
- Data: `outputs/data/activations/texts_combined.h5`

**Outputs:**
- Neutral PCs: `probes/results/neutral_pcs/layer_{N}_neutral_pcs_k20.npy`
- Config: `probes/results/neutral_pcs/config_k20.json`
- Log: `probes/logs/neutral_pcs_99300.log`

### 3. Full Lambda Sweep (ID: 99301)
**Status:** Pending (waiting for job 99300)
**Script:** `probes/scripts/slurm_jobs/training/train_orthogonal_regularized_sweep.sh`
**Configuration:**
- Layers: 10, 15, 20, 25, 30
- Lambdas: 0.0, 0.01, 0.1, 1.0, 10.0, 100.0
- Total jobs: 30 (6 lambdas × 5 layers)
- Max concurrent: 10

**Outputs:**
- Probes: `probes/results/orthogonal_regularized_probes/probe_layer{N}_all_ortho_k20_lambda{λ}.pkl`
- Summaries: `probes/results/orthogonal_regularized_probes/*_summary.txt`
- Logs: `probes/logs/ortho_reg_probe_99301_*.log`

---

## Monitoring Progress

### Check Job Status
```bash
squeue -u $USER -n ortho_test,neutral_pcs,ortho_reg_probe
```

### Run Progress Script
```bash
bash probes/scripts/slurm_jobs/utils/check_orthogonal_regularized_progress.sh
```

### View Logs (Real-time)
```bash
# Test job
tail -f probes/logs/ortho_test_99299.log

# Neutral PCs computation
tail -f probes/logs/neutral_pcs_99300.log

# Training sweep (once started)
tail -f probes/logs/ortho_reg_probe_99301_*.log
```

### Check Specific Results
```bash
# Test baseline summary
cat probes/results/orthogonal_regularized_probes_test/baseline/*_summary.txt

# Test orthogonal summary
cat probes/results/orthogonal_regularized_probes_test/lambda_1.0/*_summary.txt

# Check neutral PCs config
cat probes/results/neutral_pcs/config_k20.json | python -m json.tool
```

---

## What to Look For

### 1. Test Job Results

**Expected:**
- Baseline probe: ~70-80% test accuracy, high neutral fraction (0.3-0.5)
- Orthogonal probe (λ=1.0): Slightly lower accuracy (~1-3% drop), much lower neutral fraction (~0.1-0.2)

**Key metrics:**
```
Test accuracy: {0.7000-0.8000}
Normalized overlap: {0.0-1.0, lower is better}
Neutral fraction: {0.0-1.0, lower is better}
```

### 2. Neutral PCs

**Check:**
- 5 layers computed (10, 15, 20, 25, 30)
- Each has shape [hidden_dim, 20]
- Config file exists with diagnostic info (mean ratios, variance explained)

### 3. Full Sweep Results

**Expected pattern:**
| Lambda | Test Acc | Neutral Fraction | Interpretation |
|--------|----------|------------------|----------------|
| 0.0    | High     | High (0.3-0.5)  | Baseline - uses neutral freely |
| 0.01   | High     | Medium (0.2-0.4)| Slight regularization |
| 0.1    | High     | Lower (0.15-0.3)| Moderate regularization |
| 1.0    | Slightly↓| Low (0.1-0.2)   | **Sweet spot** |
| 10.0   | Lower    | Very low (<0.1) | Strong regularization |
| 100.0  | Much↓    | ~0              | Over-regularized |

**Good result:** λ=1.0 or λ=10.0 with <3% accuracy drop and >50% reduction in neutral fraction.

---

## Troubleshooting

### Job Stuck/Failed

**Check logs for errors:**
```bash
grep -i "error\|failed\|exception" probes/logs/*.err
```

**Common issues:**
1. Out of memory → Reduce batch size in scripts
2. Data file not found → Check path in script
3. GPU issues → Check `nvidia-smi` on compute node

**Cancel and restart:**
```bash
scancel <job_id>
sbatch probes/scripts/slurm_jobs/training/test_orthogonal_regularized_single.sh
```

### No Results After Job Completes

**Check if job succeeded:**
```bash
sacct -j <job_id> --format=JobID,State,ExitCode
```

**Check output directories:**
```bash
ls -lh probes/results/neutral_pcs/
ls -lh probes/results/orthogonal_regularized_probes/
```

### Accuracy Drops Too Much

**Solutions:**
1. Reduce λ (try 0.1 or 0.01)
2. Reduce k (fewer neutral PCs to orthogonalize against)
3. Increase weight_decay (more L2 regularization)

---

## Next Steps After Jobs Complete

### 1. Analyze Results

Create analysis script:
```python
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load all results
results = {}
for layer in [10, 15, 20, 25, 30]:
    results[layer] = {}
    for lambda_val in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
        path = f"probes/results/orthogonal_regularized_probes/probe_layer{layer}_all_ortho_k20_lambda{lambda_val}.pkl"
        try:
            results[layer][lambda_val] = pickle.load(open(path, "rb"))
        except FileNotFoundError:
            pass

# Create comparison DataFrame
data = []
for layer in results:
    for lambda_val in results[layer]:
        r = results[layer][lambda_val]
        data.append({
            'layer': layer,
            'lambda': lambda_val,
            'test_acc': r['test_accuracy'],
            'ortho_norm': r['ortho_metrics']['ortho_normalized'],
            'neutral_frac': r['ortho_metrics']['neutral_fraction']
        })

df = pd.DataFrame(data)
print(df.to_string())

# Plot accuracy vs orthogonality trade-off
for layer in [10, 15, 20, 25, 30]:
    layer_df = df[df['layer'] == layer]
    plt.scatter(layer_df['ortho_norm'], layer_df['test_acc'], label=f'Layer {layer}')

plt.xlabel('Orthogonality Violation (normalized)')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title('Accuracy vs Orthogonality Trade-off')
plt.savefig('orthogonal_probes_pareto.png')
```

### 2. Compare to Baseline Methods

**Compare to:**
- Standard linear probes (λ=0.0)
- Your existing cPCA-based probes
- Your ratio-based PCA probes

**Metrics:**
- Test accuracy (should be similar)
- Orthogonality to neutral (should be much better)
- Downstream steering performance

### 3. Test on Steering Tasks

**Use trained probes for steering:**
```python
# Load orthogonal probe
probe = results['model']
probe_weights = probe.linear.weight  # [n_classes, hidden_dim]

# Use as steering vectors (similar to your frustration_steering)
# Test if orthogonality improves steering quality
```

### 4. If Results Look Good: Scale Up

**Options:**
1. Train on all layers (0-31)
2. Try different k values (10, 50, 100)
3. Test on conversation data (user/assistant split)
4. Combine with your multi-orthogonal probes

---

## Files Created

### Implementation Files
1. `probes/methods/orthogonal_regularized_probe.py` - Core implementation
2. `probes/scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py` - Neutral PC computation
3. `probes/scripts/training/train_orthogonal_regularized_probe.py` - Training script

### SLURM Scripts
4. `probes/scripts/slurm_jobs/dimensionality_reduction/compute_neutral_pcs.sh`
5. `probes/scripts/slurm_jobs/training/test_orthogonal_regularized_single.sh`
6. `probes/scripts/slurm_jobs/training/train_orthogonal_regularized_sweep.sh`

### Utilities
7. `probes/scripts/slurm_jobs/utils/check_orthogonal_regularized_progress.sh` - Progress monitoring
8. `probes/methods/README_ORTHOGONAL_REGULARIZATION.md` - Full documentation
9. This file - Run summary

---

## Key Innovation

**Loss function:**
```
L_total = L_emotion + λ * ||U^T @ w||²
```

Where:
- `U` = neutral PCs [hidden_dim, k] - pre-computed neutral-dominated directions
- `w` = probe weights [n_classes, hidden_dim]
- `λ` = orthogonality penalty weight

**Goal:** Learn emotion representations orthogonal to neutral content structure.

**Expected benefit:** Better disentanglement of emotion from content, potentially improving steering and interpretability.

---

## Contact & Questions

Check logs and results as they complete. If you see unexpected behavior or have questions about the implementation, the detailed documentation is in:
- `probes/methods/README_ORTHOGONAL_REGULARIZATION.md`

Good luck with the experiments! 🚀
