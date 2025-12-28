# Multi-Orthogonal Emotion Probes: Discovering Intrinsic Dimensionality

## Overview

This experiment trains **K orthogonal emotion probe sets** on text data to discover the intrinsic dimensionality of emotion representations. Unlike conversation probes (which have 2 fixed sets for user/assistant), this approach:

1. **Searches for optimal K**: Incrementally tests K=1,2,3,... until convergence fails
2. **General emotion probes**: Not tied to speaker types - all sets classify the same 6 emotions
3. **All-pairs orthogonality**: Each of K sets is constrained to be orthogonal to all other sets
4. **Dimensionality discovery**: The maximum K that converges indicates the number of independent emotion subspaces

## Key Idea

**Question**: How many independent ways can a model represent emotions?

If we can successfully train K orthogonal probe sets that each achieve good classification accuracy, this suggests the representation contains at least K independent emotion subspaces. When K becomes too large, either:
- Accuracy degrades (not enough independent information)
- Orthogonality won't converge (the subspaces interfere)

This reveals the **intrinsic dimensionality** of emotion encoding.

## Architecture

### Model: `MultiOrthogonalEmotionProbes`

```python
class MultiOrthogonalEmotionProbes(nn.Module):
    def __init__(self, hidden_dim, n_emotions=6, n_sets=K):
        # All K probe sets
        self.probe_sets = nn.Parameter(torch.randn(K, n_emotions, hidden_dim))
```

**Parameters**: `[K, n_emotions, hidden_dim]`
- K independent probe sets
- Each set has 6 emotion directions
- All trained simultaneously

### Loss Function

```
L_total = L_task + λ × L_ortho
```

#### Task Loss (Classification)
Average cross-entropy across all K sets:

```python
L_task = (1/K) × Σ_{k=1}^K CrossEntropy(logits_k, labels)
```

Each set independently classifies emotions. All sets trained on the same data.

#### Orthogonality Loss (All-Pairs Constraint)
Penalizes dot products between **all pairs** of sets:

```python
L_ortho = mean over all (i,j) pairs where i<j:
    mean((probe_set_i @ probe_set_j.T)²)
```

**Number of constraints**: O(K²) pairs
- K=2: 1 pair
- K=3: 3 pairs
- K=5: 10 pairs
- K=10: 45 pairs

As K increases, maintaining orthogonality becomes harder.

## Metrics

### Per-Set Metrics
- **Accuracies**: [acc_1, acc_2, ..., acc_K] for each set
- **Mean accuracy**: Average across all K sets
- **Std accuracy**: Variance between sets
- **Min/max accuracy**: Range of set performances

### Independence Metrics
- **Cross-set agreement**: How often sets agree on predictions (should be LOW)
  - If truly independent: ~16.7% (random agreement for 6 classes)
  - If correlated: Higher agreement

- **Orthogonality**: Mean |dot product| between all pairs
  - Goal: < 0.1 (nearly orthogonal)
  - If fails to converge: Representation can't support K independent subspaces

### Convergence Criteria

A configuration with K sets is considered **converged** if:
1. **Orthogonality**: `mean |cross-dot| < 0.1`
2. **Accuracy**: `mean accuracy > 0.25` (well above random 16.7%)

## Search Procedure

The `--search-k` mode incrementally tests K=1,2,3,...,max_k and stops when:

1. **Convergence failure**: Orthogonality loss won't converge below threshold
2. **Accuracy degradation**: Mean accuracy drops > 5% from previous K
3. **Max K reached**: Tested all values up to max_k

### Example Search Results

```
K=1: mean_acc=0.65, ortho=0.00 ✓ (no orthogonality needed)
K=2: mean_acc=0.63, ortho=0.03 ✓ (good separation)
K=3: mean_acc=0.61, ortho=0.05 ✓ (still converges)
K=4: mean_acc=0.58, ortho=0.08 ✓ (marginal)
K=5: mean_acc=0.52, ortho=0.15 ✗ (orthogonality failed)
```

**Interpretation**: The representation supports **4 independent emotion subspaces**.

This suggests the intrinsic dimensionality for emotion is >= 4×6 = 24 dimensions.

## Usage

### 1. Search for Optimal K

```bash
# Search up to K=10 at layer 30
sbatch probes/scripts/slurm_jobs/training/run_multi_orthogonal_search.sh 30 10 10.0

# Or run directly
python probes/scripts/training/train_multi_orthogonal_text_probes.py \
    --layer 30 \
    --search-k \
    --max-sets 10 \
    --ortho-weight 10.0
```

**Output**: `outputs/probes/emotion_probes/text_based/multi_orthogonal/search_layer30_ortho10.0_seed42.pkl`

### 2. Train Fixed K

```bash
# Train with K=3 probe sets
python probes/scripts/training/train_multi_orthogonal_text_probes.py \
    --layer 30 \
    --n-sets 3 \
    --ortho-weight 10.0
```

**Output**: `outputs/probes/emotion_probes/text_based/multi_orthogonal/probe_k3_layer30_ortho10.0.pkl`

### 3. Analyze Results

```bash
python probes/scripts/visualization/analyze_multi_orthogonal_results.py \
    --search-results outputs/probes/.../search_layer30_ortho10.0_seed42.pkl \
    --output results/multi_ortho_analysis/
```

**Outputs**:
- `k_vs_metrics.png`: How accuracy/orthogonality change with K
- `convergence_curves.png`: Training dynamics for each K
- `similarity_matrices.png`: Inter-set similarity heatmaps
- Console: Summary table and dimensionality analysis

## Hyperparameters

### Critical Parameters

**`--ortho-weight` (λ)**: Orthogonality penalty weight
- **Low (1.0-5.0)**: Prioritizes accuracy, may not achieve full orthogonality
- **Medium (5.0-20.0)**: ← **Recommended** balanced tradeoff
- **High (20.0-100.0)**: Strict orthogonality, may reduce accuracy

**`--max-sets`**: Maximum K to test in search
- Start with 10, increase if all converge
- Computation: O(K²) constraints, so K=10 is ~45 pairs

**`--convergence-threshold`**: Max mean |cross-dot| for convergence
- Default: 0.1 (pretty orthogonal)
- Lower: Stricter (may fail earlier)
- Higher: More lenient (may overestimate K)

**`--accuracy-degradation-threshold`**: Max accuracy drop K→K+1
- Default: 0.05 (5% drop)
- Stop search if accuracy drops more than this

### Training Parameters

```bash
--max-epochs 200           # Usually converges in 50-150 epochs
--patience 20              # Early stopping patience
--batch-size 64            # Larger batch = more stable gradients
--learning-rate 0.001      # Adam default works well
```

## Interpretation Guide

### What K Tells Us

**K=1**: Baseline - single probe set (no orthogonality needed)
- Should achieve ~60-65% accuracy on 6-class emotion

**K=2-4**: Typical range for emotion representations
- Suggests 2-4 independent emotion subspaces
- Total dimensionality: K × 6 emotions

**K>5**: Surprising if this converges!
- Would suggest very high-dimensional emotion encoding
- Or the representation is over-parameterized

### Accuracy Patterns

**All sets similar accuracy** (std ~0.02):
- Good sign - all sets learning equally well
- Suggests balanced orthogonal subspaces

**High variance** (std >0.1):
- Some sets learning, others not
- May indicate uneven subspace structure
- Or K is too high

**Decreasing mean accuracy with K**:
- Expected - harder to maintain K orthogonal sets
- Sharp drop indicates hitting capacity limit

### Cross-Set Agreement

**Low agreement** (~16-20%):
- ✓ Sets are making independent predictions
- Confirms orthogonality at decision level

**High agreement** (>30%):
- ⚠ Sets are correlated despite orthogonality constraint
- May share common features
- Consider increasing ortho_weight

## Advanced: With cPCA

You can also search for K on dimensionality-reduced representations:

```bash
python probes/scripts/training/train_multi_orthogonal_text_probes.py \
    --layer 30 \
    --cpca-path outputs/dimensionality_reduction/cpca/text_based/google/gemma-3-27b-it_cpca.npz \
    --n-components 20 \
    --search-k \
    --max-sets 10 \
    --ortho-weight 5.0
```

**Question**: Does cPCA (which captures emotion-relevant variance) support fewer or more orthogonal subspaces than raw activations?

## Comparison to Conversation Probes

| Aspect | Conversation Probes | Multi-Orthogonal Probes |
|--------|-------------------|------------------------|
| **Purpose** | Separate user/assistant | Discover dimensionality |
| **K value** | Fixed K=2 | Search for optimal K |
| **Data** | Conversation pairs | Text labels only |
| **Labels** | (user_emotion, asst_emotion) | Single emotion per sample |
| **Constraint** | User ⊥ Assistant | All pairs orthogonal |
| **Output** | Speaker-specific probes | K independent probe sets |

## Research Questions

1. **How many independent emotion subspaces exist?**
   - Run search at multiple layers
   - Compare raw vs cPCA representations

2. **Does K vary by layer?**
   - Early layers: Lower K? (basic features)
   - Late layers: Higher K? (complex representations)

3. **What do the K sets capture?**
   - Qualitative analysis: What does each set specialize in?
   - Do some sets focus on specific emotion clusters?

4. **Relation to other metrics?**
   - Does K correlate with probe accuracy?
   - Does K relate to model size/depth?

## Example Output

```
================================================================================
SEARCHING FOR OPTIMAL K (max_sets=8)
================================================================================

Training with K=1 probe sets, ortho_weight=10.0
...
✓ K=1 converged successfully with mean accuracy 0.6523

Training with K=2 probe sets, ortho_weight=10.0
...
✓ K=2 converged successfully with mean accuracy 0.6341

Training with K=3 probe sets, ortho_weight=10.0
...
✓ K=3 converged successfully with mean accuracy 0.6128

Training with K=4 probe sets, ortho_weight=10.0
...
✓ K=4 converged successfully with mean accuracy 0.5847

Training with K=5 probe sets, ortho_weight=10.0
...
⚠ K=5 did not converge. Stopping search.

================================================================================
SUMMARY TABLE
================================================================================
K    Mean Acc     Std Acc      Ortho Loss   Cross-Agree  Converged
--------------------------------------------------------------------------------
1    0.6523       0.0000       0.0000       0.0000       ✓
2    0.6341       0.0123       0.0287       0.1752       ✓
3    0.6128       0.0156       0.0431       0.1823       ✓
4    0.5847       0.0234       0.0789       0.1891       ✓
5    0.5234       0.0421       0.1523       0.2134       ✗
================================================================================

✓ Best K: 4 with mean accuracy 0.5847

Maximum converged K: 4
  (with accuracy >= 30.0%)

Interpretation:
  The representation supports at least 4 independent
  emotion subspaces, suggesting the intrinsic dimensionality for
  emotion encoding is >= 24 (K × n_emotions).

  Note: This is a lower bound - the true dimensionality may be higher.
```

## Files

### Training
- `probes/scripts/training/train_multi_orthogonal_text_probes.py` - Main training script
- `probes/scripts/slurm_jobs/training/run_multi_orthogonal_search.sh` - SLURM launcher

### Analysis
- `probes/scripts/visualization/analyze_multi_orthogonal_results.py` - Visualization and analysis

### Output Structure
```
outputs/probes/emotion_probes/text_based/multi_orthogonal/
├── search_layer30_ortho10.0_seed42.pkl          # Full search results
├── best_k4_layer30_ortho10.0.pkl                # Best model from search
└── probe_k3_layer30_ortho10.0.pkl               # Single fixed-K training
```

## Tips

1. **Start with moderate ortho_weight** (10.0) and adjust if needed
2. **Check convergence curves** - if oscillating, reduce learning rate
3. **Compare across layers** - does K increase in deeper layers?
4. **Use larger batch sizes** (64-128) for more stable gradients with many sets
5. **If all K converge**, increase max_sets to find the limit
6. **If none converge**, reduce ortho_weight or increase convergence threshold

## Citation

If you use this method, please cite the original orthogonal probe training approach and note this extension to K>2 sets for dimensionality discovery.
