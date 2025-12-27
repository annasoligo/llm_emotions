# Unified Output Directory Structure

This document describes the unified output organization for all research-tools results.

## Overview

All outputs are organized under a single **`outputs/`** directory with clear categorical separation. This replaces the previous fragmented structure with results scattered across `results/` and `probes/results/`.

## Directory Structure

```
outputs/
├── data/                           # Intermediate activation data
├── dimensionality_reduction/       # cPCA, PCA, Ratio PCA results
├── probes/                         # Trained emotion & conversation probes
├── evaluations/                    # Evaluation results
├── interpretations/                # PC interpretations & analysis
└── visualizations/                 # All plots and figures
```

## Detailed Organization

### 1. Data (`outputs/data/`)

Intermediate activation data from model inference.

```
outputs/data/
├── activations/
│   ├── texts_combined.h5              # Combined tier text activations
│   ├── conversations_combined.h5       # Combined conversation activations
│   ├── conversations_oracle_filtered.h5  # Oracle-filtered conversations
│   └── regional/                      # Regional activation splits
│       ├── user_turns.h5
│       ├── assistant_turns.h5
│       ├── special_tokens_1.h5
│       └── special_tokens_2.h5
└── README.md
```

**Scripts that write here:**
- `probes/scripts/data_collection/collect_activations.py`
- `probes/scripts/data_collection/combine_activations.py`
- `probes/scripts/data_collection/filter_conversations_with_oracle.py`
- `probes/scripts/data_collection/extract_regional_from_combined.py`

---

### 2. Dimensionality Reduction (`outputs/dimensionality_reduction/`)

cPCA and Ratio PCA component analysis results.

```
outputs/dimensionality_reduction/
├── cpca/
│   ├── tier_based/
│   │   ├── google/
│   │   │   ├── gemma-3-27b-it_cpca.npz           # cPCA components
│   │   │   └── gemma-3-27b-it_cpca_metadata.json # Hyperparameters
│   │   ├── alpha_sweep/
│   │   │   ├── alpha_1.0/
│   │   │   ├── alpha_5.0/  (recommended)
│   │   │   └── alpha_10.0/
│   │   └── README.md
│   │
│   ├── conversation_based/
│   │   ├── global/                    # Global conversation activations
│   │   │   └── google/
│   │   │       └── gemma-3-27b-it_cpca.npz
│   │   └── regional/                  # Regional splits
│   │       ├── user/
│   │       ├── assistant/
│   │       ├── special_tokens_1/
│   │       └── special_tokens_2/
│   │
│   └── combined_ranges/               # Merged layer ranges
│       └── gemma-3-27b-it_layers_0_60.npz
│
├── ratio_pca/
│   ├── tier_based/
│   │   ├── k5/
│   │   ├── k10/
│   │   ├── k30/
│   │   └── k_tuning/                  # Parameter sweep
│   │       ├── tuning_results.json
│   │       └── accuracy_by_k.png
│   └── README.md
│
└── comparisons/
    ├── cpca_vs_ratio_pca_k5.json
    ├── cpca_vs_ratio_pca_k30.json
    └── method_comparison_plots/
        ├── accuracy_comparison.png
        └── component_similarity.png
```

**Scripts that write here:**
- `probes/scripts/dimensionality_reduction/run_cpca.py`
- `probes/scripts/dimensionality_reduction/run_cpca_layer_range.py`
- `probes/scripts/dimensionality_reduction/run_regional_cpca.py`
- `probes/scripts/dimensionality_reduction/run_ratio_pca.py`
- `probes/scripts/dimensionality_reduction/tune_ratio_pca_k.py`
- `probes/scripts/dimensionality_reduction/combine_cpca_layer_ranges.py`
- `probes/scripts/evaluation/compare_pca_methods.py`

---

### 3. Probes (`outputs/probes/`)

Trained linear probe models for emotion classification.

```
outputs/probes/
├── emotion_probes/
│   ├── text_based/
│   │   ├── raw/                       # Raw activation probes
│   │   │   ├── probe_layer10_all_raw.pkl
│   │   │   ├── probe_layer10_all_raw_summary.txt
│   │   │   └── ...
│   │   │
│   │   ├── cpca/                      # cPCA-projected probes
│   │   │   ├── top3/
│   │   │   ├── top5/
│   │   │   ├── top10/ (recommended)
│   │   │   └── top20/
│   │   │       ├── probe_layer10_all_cpca_top20.pkl
│   │   │       └── probe_layer10_all_cpca_top20_summary.txt
│   │   │
│   │   ├── regularization/
│   │   │   ├── l1/
│   │   │   │   └── probe_layer*_l1_*.pkl
│   │   │   └── high_alpha_cpca/
│   │   │       └── probe_layer*_cpca_high_alpha.pkl
│   │   │
│   │   ├── multiseed/                 # Multi-seed robustness
│   │   │   ├── seed_0/
│   │   │   ├── seed_42/
│   │   │   ├── seed_100/
│   │   │   └── seed_200/
│   │   │
│   │   └── README.md
│   │
│   └── conversation_based/
│       ├── standard/                  # Standard conversation probes
│       │   ├── probe_layer10_user_global_cpca_top10.pkl
│       │   ├── probe_layer10_asst_global_cpca_top10.pkl
│       │   └── overlays/              # Overlayed accuracy plots
│       │
│       ├── orthogonal/                # Orthogonal user ⊥ assistant probes
│       │   ├── ortho_1.0/
│       │   ├── ortho_10.0/
│       │   ├── ortho_100.0/ (recommended)
│       │   └── ortho_1000.0/
│       │       ├── raw/
│       │       ├── global_cpca_top10/
│       │       └── regional_cpca_top10/
│       │
│       └── README.md
│
└── manifests/
    ├── all_probes_summary.json        # Complete probe inventory
    ├── accuracy_by_layer.png          # Layer-wise accuracy plot
    └── best_configs.json              # Top-performing configurations
```

**Scripts that write here:**
- `probes/scripts/training/train_emotion_probe.py`
- `probes/scripts/training/train_emotion_probe_multiseed.py`
- `probes/scripts/training/train_conversation_probe.py`
- `probes/scripts/training/train_orthogonal_conversation_probe.py`

---

### 4. Evaluations (`outputs/evaluations/`)

Evaluation results for trained probes.

```
outputs/evaluations/
├── conversation_eval/
│   ├── text_based_probes/
│   │   ├── raw/
│   │   │   └── eval_results_layer*_raw.json
│   │   ├── cpca_variants/
│   │   │   ├── top3/
│   │   │   ├── top5/
│   │   │   ├── top10/
│   │   │   └── top20/
│   │   └── multiseed/
│   │       ├── eval_multiseed_shuffled.json
│   │       └── eval_multiseed_filtered.json
│   │
│   ├── conversation_based_probes/
│   │   ├── standard/
│   │   ├── orthogonal/
│   │   └── regional_variants/
│   │
│   └── plots/
│       ├── accuracy_heatmaps/
│       ├── dimensionality_analysis/
│       └── multiseed_comparisons/
│
├── emo_lens_experiments/
│   ├── single_layer/
│   │   └── {question_module}/         # e.g., vertex_helios, value_persistence
│   │       ├── results.json
│   │       └── visualizations/
│   │           ├── heatmaps/
│   │           └── trajectories/
│   │
│   └── multilayer/
│       └── {question_module}/
│           ├── results.json           # 900KB detailed results
│           └── heatmaps/
│
├── comparative_analysis/
│   ├── pca_method_comparison/
│   │   ├── cpca_vs_ratio_pca_k5/
│   │   ├── cpca_vs_ratio_pca_k30/
│   │   └── comparison_plots/
│   │
│   ├── regularization_comparison/
│   │   ├── l1_vs_l2.json
│   │   └── regularization_comparison.png
│   │
│   └── autointerp_comparison/
│       └── interpretation_alignment.json
│
└── README.md
```

**Scripts that write here:**
- `probes/scripts/evaluation/eval_probes_on_conversations.py`
- `probes/scripts/evaluation/eval_multiseed_probes.py`
- `probes/scripts/evaluation/eval_multiseed_fast.py`
- `probes/scripts/evaluation/run_emo_lens_probe_experiment.py`
- `probes/scripts/evaluation/compare_pca_methods.py`
- `probes/scripts/evaluation/compare_regularization_methods.py`
- `probes/scripts/evaluation/compare_autointerp_results.py`

---

### 5. Interpretations (`outputs/interpretations/`)

Automated interpretation of principal components and probe weights.

```
outputs/interpretations/
├── autointerp/
│   ├── tier_based/
│   │   ├── third_person/
│   │   │   ├── layer_10.json
│   │   │   ├── layer_20.json
│   │   │   └── ...
│   │   ├── second_person_eliciting/
│   │   └── direct_address/
│   │
│   ├── conversation_based/
│   │   ├── global/
│   │   ├── regional_user/
│   │   ├── regional_assistant/
│   │   ├── regional_special1/
│   │   └── regional_special2/
│   │
│   └── ratio_pca/
│       ├── k10/
│       └── k30/
│
├── pc_analysis/
│   ├── weight_distributions/
│   │   ├── layer_10_weights.png
│   │   ├── layer_30_weights.png
│   │   └── all_layers_summary.png
│   │
│   ├── component_meanings/
│   │   ├── pc_meanings_and_weights_layer10.txt
│   │   ├── pc_meanings_and_weights_layer30.txt
│   │   └── combined_analysis.json
│   │
│   └── semantic_alignment/
│       └── emotion_axis_analysis.json
│
└── README.md
```

**Scripts that write here:**
- `probes/scripts/evaluation/autointerp_pcs.py`
- `probes/scripts/evaluation/analyze_pc_meanings_and_weights.py`
- `probes/scripts/visualization/visualize_pc_weights.py`

---

### 6. Visualizations (`outputs/visualizations/`)

All plots, heatmaps, and figures.

```
outputs/visualizations/
├── probe_performance/
│   ├── accuracy_by_layer/
│   │   ├── text_probes_all_layers.png
│   │   └── conversation_probes_all_layers.png
│   │
│   ├── accuracy_heatmaps/
│   │   ├── layer_representation_heatmap.png
│   │   └── multiseed_accuracy_heatmap.png
│   │
│   └── dimensionality_analysis/
│       ├── accuracy_vs_n_components.png
│       └── regularization_effects.png
│
├── conversation_analysis/
│   ├── user_vs_assistant/
│   │   ├── overlayed_accuracy.png
│   │   └── emotion_distributions.png
│   │
│   └── orthogonal_probes/
│       ├── ortho_weight_comparison.png
│       ├── regional_importance.png
│       └── accuracy_by_ortho_weight.png
│
├── component_analysis/
│   ├── pc_weight_distributions/
│   ├── component_importance/
│   └── explained_variance/
│
├── emotion_trajectories/
│   ├── emo_lens_heatmaps/
│   ├── layer_trajectories/
│   └── comparative_trajectories/
│
└── method_comparisons/
    ├── cpca_vs_ratio_pca/
    ├── regularization_effects/
    └── representation_quality/
```

**Scripts that write here:**
- `probes/scripts/visualization/plot_probe_results.py`
- `probes/scripts/visualization/plot_layer_accuracy.py`
- `probes/scripts/visualization/plot_conversation_probe_accuracy.py`
- `probes/scripts/visualization/plot_conversation_overlayed_accuracy.py`
- `probes/scripts/visualization/plot_conversation_user_asst_accuracy.py`
- `probes/scripts/visualization/plot_orthogonal_probe_results.py`
- `probes/scripts/visualization/plot_ortho_weight_comparison.py`
- `probes/scripts/visualization/plot_orthogonal_regional_importance.py`
- `probes/scripts/visualization/plot_regional_weight_importance.py`
- `probes/scripts/visualization/visualize_conversation_eval.py`
- `probes/scripts/visualization/visualize_all_conversation_eval.py`
- `probes/scripts/visualization/visualize_multiseed_results.py`
- `probes/scripts/visualization/visualize_pc_weights.py`
- `probes/scripts/visualization/generate_all_visualizations.py`

---

## Migration from Old Structure

### Old → New Mapping

| Old Path | New Path |
|----------|----------|
| `results/emotion_probes/` | `outputs/probes/emotion_probes/text_based/raw/` |
| `results/emotion_probes_top10/` | `outputs/probes/emotion_probes/text_based/cpca/top10/` |
| `results/emotion_probes_multiseed/` | `outputs/probes/emotion_probes/text_based/multiseed/` |
| `results/conversation_eval/` | `outputs/evaluations/conversation_eval/` |
| `results/emo_lens_probe_experiments/` | `outputs/evaluations/emo_lens_experiments/single_layer/` |
| `results/autointerp/` | `outputs/interpretations/autointerp/tier_based/` |
| `results/pc_weight_visualizations/` | `outputs/visualizations/component_analysis/pc_weight_distributions/` |
| `probes/results/cpca_tier_data/` | `outputs/dimensionality_reduction/cpca/tier_based/` |
| `probes/results/cpca_tier_data_high_alpha/` | `outputs/dimensionality_reduction/cpca/tier_based/alpha_sweep/alpha_5.0/` |
| `probes/results/cpca_conversations_global/` | `outputs/dimensionality_reduction/cpca/conversation_based/global/` |
| `probes/results/cpca_conversations_regional_*/` | `outputs/dimensionality_reduction/cpca/conversation_based/regional/*/` |
| `probes/results/conversation_probes/` | `outputs/probes/emotion_probes/conversation_based/standard/` |
| `probes/results/conversation_probes_orthogonal/` | `outputs/probes/emotion_probes/conversation_based/orthogonal/` |
| `probes/results/autointerp/` | `outputs/interpretations/autointerp/tier_based/` |

---

## Benefits of New Structure

1. **Single Root**: All outputs in one place (`outputs/`)
2. **Logical Categorization**: Clear separation by function (probes, evaluations, interpretations, visualizations)
3. **Hierarchical Organization**: Related experiments grouped together (e.g., all top-k variants under `cpca/`)
4. **Consistent Naming**: No more `_top3`, `_top5`, `_top10` directory proliferation
5. **Scalability**: Easy to add new experiment types without cluttering
6. **Discoverability**: README files in each category explain contents
7. **Metadata**: JSON manifests track experiment parameters and results

---

## Implementation Notes

### For Script Authors

When writing/updating scripts:

1. **Use relative paths from repo root:**
   ```python
   from pathlib import Path

   REPO_ROOT = Path(__file__).parent.parent.parent  # Adjust as needed
   OUTPUT_DIR = REPO_ROOT / "outputs" / "probes" / "emotion_probes" / "text_based" / "cpca" / "top10"
   ```

2. **Make output paths configurable:**
   ```python
   parser.add_argument("--output-dir", type=Path,
                       default="outputs/probes/emotion_probes/text_based/cpca/top10")
   ```

3. **Create parent directories automatically:**
   ```python
   output_path.parent.mkdir(parents=True, exist_ok=True)
   ```

4. **Write metadata files:**
   ```python
   metadata = {
       "script": __file__,
       "timestamp": datetime.now().isoformat(),
       "hyperparameters": {...},
       "model": model_name,
   }
   with open(output_dir / "metadata.json", "w") as f:
       json.dump(metadata, f, indent=2)
   ```

### For SLURM Scripts

Update job scripts to use new paths:

```bash
# Old
python -m probes.scripts.training.train_emotion_probe \
    --output-dir results/emotion_probes_top10

# New
python -m probes.scripts.training.train_emotion_probe \
    --output-dir outputs/probes/emotion_probes/text_based/cpca/top10
```

---

## Maintenance

- **Regular cleanup**: Remove experiment variants that didn't work
- **Archive old results**: Move superseded experiments to `outputs/archive/`
- **Update README files**: Keep category README files current
- **Version manifests**: Track which scripts/versions produced which results
- **Git LFS**: Consider using Git LFS for large output files (>100MB)

---

For questions or suggestions, see [probes/scripts/README.md](../probes/scripts/README.md).
