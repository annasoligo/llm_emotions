# Probes Scripts

Python scripts for the emotion probes pipeline.

## Recent Consolidation (Dec 2024)

### Python Scripts Cleanup
- **Oracle filtering**: Extracted common utilities to `oracle_filtering_utils.py` (~40% code deduplication)
- **Activation combination**: Merged 2 scripts → 1 unified `combine_activations.py` with `--type` parameter
- **Visualization scripts**: Kept separate (different purposes, no meaningful overlap)

**Result**: Reduced duplication by ~200 lines, cleaner codebase

---

## Core Pipeline

### 1. Data Generation
- **`generate_data.py`** - Generate emotional conversations/texts (Claude + Gemma)
- **`generate_neutral_conversations_batch.py`** - Neutralize conversations via Batch API

### 2. Activation Collection
- **`collect_activations.py`** - Collect neural activations (texts/conversations)
- **`collect_paired_conversation_activations.py`** - Collect paired emotional+neutral activations

### 3. Contrastive PCA & Ratio PCA
- **`run_cpca.py`** - Run cPCA from config file
- **`run_cpca_layer_range.py`** - Parallel cPCA across layer ranges
- **`run_regional_cpca.py`** - Independent regional cPCA
- **`cpca_conversations_emotion_contrast.py`** - Emotion-vs-emotion contrast
- **`run_ratio_pca.py`** - Run ratio-based PCA (PCA v3 - simpler alternative to cPCA)
- **`compare_pca_methods.py`** - Compare ratio PCA vs cPCA results

See [../methods/README_RATIO_PCA.md](../methods/README_RATIO_PCA.md) for ratio PCA details.

### 4. Auto-Interpretation
See [AUTOINTERP_README.md](AUTOINTERP_README.md) for details.
- **`autointerp_pcs.py`** - Interpret text PCs with median samples
- **`autointerp_conversation_pcs.py`** - Interpret conversation PCs with dual mode

### 5. Oracle Filtering
- **`filter_with_oracle.py`** - Filter texts with oracle (multi-turn, direct implementation)
- **`filter_conversations_with_oracle.py`** - Filter conversations with oracle (single-turn, uses oracle_helpers)
- **`oracle_filtering_utils.py`** - Shared utilities (NEW)

### 6. Activation Utilities ✨
- **`combine_activations.py`** - Unified activation combiner (NEW)
  ```bash
  # Standard mode (nested structure)
  python -m probes.scripts.combine_activations \
      --type standard \
      --emotional data/activations/conversations2.h5 \
      --neutral data/activations/conversations2_neutral.h5 \
      --output data/activations/conversations2_combined.h5

  # Regional mode (separate files per region)
  python -m probes.scripts.combine_activations \
      --type regional \
      --emotional data/activations/conversations2.h5 \
      --neutral data/activations/conversations2_neutral.h5 \
      --output-dir data/activations/regional
  ```

**Replaced**:
- ❌ `combine_conversation_activations.py` (deleted)
- ❌ `combine_conversation_activations_regional.py` (deleted)

### 7. Probe Training & Evaluation
- **`train_emotion_probe.py`** - Train multiclass emotion probes
- **`eval_probes_on_conversations.py`** - Evaluate probes on conversations
- **`test_emotion_steering.py`** - Test emotion steering

### 8. Visualization
- **`visualize_conversation_eval.py`** - Single-result analysis (confusion matrices, per-emotion accuracy)
- **`visualize_all_conversation_eval.py`** - Multi-result comparison (PC count effects, heatmaps)
- **`visualize_pc_weights.py`** - PC weight distributions
- **`generate_all_visualizations.py`** - Batch visualization generation

**Note**: Visualization scripts kept separate - they serve different purposes with no meaningful code overlap.

### 9. Analysis Utilities
- **`analyze_pc_meanings_and_weights.py`** - PC semantic content analysis
- **`compare_regularization_methods.py`** - Regularization comparison
- **`plot_layer_accuracy.py`** - Layer accuracy plots
- **`conversation_eval_utils.py`** - Shared evaluation utilities
- **`extract_activation_view.py`** - Extract specific activation views
- **`extract_regional_from_combined.py`** - Extract regional data
- **`fix_conversation_metadata.py`** - One-off metadata fixes

---

## SLURM Jobs

See [slurm_jobs/README.md](slurm_jobs/README.md) for complete SLURM job documentation.

**Key scripts**:
- `run_cpca.sh` - Generic cPCA runner
- `eval_probes.sh` - Generic probe evaluator (single/sweep modes)
- `run_autointerp.sh` - Generic PC interpreter

---

## Archive

Deprecated scripts in [archive/](archive/):
- `analyze_cpca_probe_weights.py`
- `plot_dimensionality_sweep.py`
- `plot_dimensionality_sweep_v2.py`

---

## Migration Guide

### Activation Combination Scripts

**Before**:
```bash
# Standard
python -m probes.scripts.combine_conversation_activations \
    --emotional data/activations/conversations2.h5 \
    --neutral data/activations/conversations2_neutral.h5 \
    --output data/activations/conversations2_combined.h5

# Regional
python -m probes.scripts.combine_conversation_activations_regional \
    --emotional data/activations/conversations2.h5 \
    --neutral data/activations/conversations2_neutral.h5 \
    --output-dir data/activations/regional
```

**After**:
```bash
# Standard
python -m probes.scripts.combine_activations \
    --type standard \
    --emotional data/activations/conversations2.h5 \
    --neutral data/activations/conversations2_neutral.h5 \
    --output data/activations/conversations2_combined.h5

# Regional
python -m probes.scripts.combine_activations \
    --type regional \
    --emotional data/activations/conversations2.h5 \
    --neutral data/activations/conversations2_neutral.h5 \
    --output-dir data/activations/regional
```

---

## Summary

**Before consolidation**: ~50 Python scripts
**After consolidation**: ~49 Python scripts + 1 shared utilities module

**Benefits**:
- ✅ ~200 lines of code deduplicated
- ✅ Cleaner, more maintainable codebase
- ✅ Single source of truth for common functionality
- ✅ Consistent interfaces across similar operations
- ✅ Better organized and documented

**Files changed**: 8
- Created: 2 (combine_activations.py, oracle_filtering_utils.py)
- Modified: 4 (filter_with_oracle.py, filter_conversations_with_oracle.py, 3 external callers)
- Deleted: 2 (old activation combination scripts)
