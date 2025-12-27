# Evaluation Results

This directory contains evaluation results for trained emotion probes on conversation data and comparative analyses.

## Directory Structure

```
evaluations/
├── conversation_eval/           # Probe performance on conversations
│   ├── text_based_probes/       # Text-based probes evaluated on conversations
│   │   ├── raw/
│   │   ├── cpca_variants/       # Different top-k configurations
│   │   └── multiseed/           # Multi-seed robustness analysis
│   ├── conversation_based_probes/  # Native conversation probes
│   └── plots/                   # Accuracy heatmaps & visualizations
│
├── emo_lens_experiments/        # Emotion trajectory analysis
│   ├── single_layer/            # Single-layer probe experiments
│   │   └── {question_module}/   # e.g., vertex_helios, value_persistence
│   │       ├── results.json
│   │       └── visualizations/
│   └── multilayer/              # Multi-layer trajectory analysis
│
└── comparative_analysis/        # Method comparison results
    ├── pca_method_comparison/   # cPCA vs Ratio PCA
    ├── regularization_comparison/  # L1, L2, etc.
    └── autointerp_comparison/   # Interpretation quality
```

## File Formats

### Evaluation Results (.json)

Large JSON files (1-2.5 MB) containing detailed per-conversation predictions:

```json
{
    "metadata": {
        "probe_path": "outputs/probes/...",
        "conversations_path": "data/conversations2.jsonl",
        "layer": 30,
        "timestamp": "2025-12-27T10:30:00"
    },
    "overall_metrics": {
        "user_accuracy": 0.942,
        "assistant_accuracy": 0.938,
        "macro_f1": 0.935
    },
    "per_emotion_metrics": {
        "happiness": {"precision": 0.95, "recall": 0.94, "f1": 0.945},
        "sadness": {...},
        ...
    },
    "predictions": [
        {
            "conversation_id": "conv_001",
            "user_emotion": "happiness",
            "user_predicted": "happiness",
            "user_correct": true,
            "assistant_emotion": "empathy",
            "assistant_predicted": "empathy",
            "assistant_correct": true,
            "confidence_scores": {...}
        },
        ...
    ]
}
```

## Usage

### Evaluating Probes on Conversations

**Single probe evaluation:**

```bash
python -m probes.scripts.evaluation.eval_probes_on_conversations \
    --probe-dir outputs/probes/emotion_probes/text_based/cpca/top10 \
    --conversations data/conversations2.jsonl \
    --layers 30 \
    --output outputs/evaluations/conversation_eval/text_based_probes/cpca_variants/top10/eval_results_layer30.json
```

**Multi-seed evaluation:**

```bash
python -m probes.scripts.evaluation.eval_multiseed_fast \
    --probe-dirs outputs/probes/emotion_probes/text_based/multiseed/seed_{0,42,100,200} \
    --conversations data/conversations2.jsonl \
    --layers 30 \
    --output outputs/evaluations/conversation_eval/text_based_probes/multiseed/eval_multiseed.json
```

### Emo Lens Experiments

Analyze emotion trajectories through model layers:

```bash
python -m probes.scripts.evaluation.run_emo_lens_probe_experiment \
    --question-module vertex_helios \
    --layers 10 20 30 40 50 \
    --output-dir outputs/evaluations/emo_lens_experiments/single_layer \
    --probe-base-path outputs/probes/emotion_probes/text_based/cpca/top10
```

**Multilayer analysis:**

```bash
python -m probes.scripts.evaluation.run_emo_lens_probe_experiment \
    --question-module vertex_helios \
    --multilayer \
    --output-dir outputs/evaluations/emo_lens_experiments/multilayer
```

### Comparative Analysis

**Compare PCA methods:**

```bash
python -m probes.scripts.evaluation.compare_pca_methods \
    --cpca-results outputs/dimensionality_reduction/cpca/tier_based/google/gemma-3-27b-it_cpca.npz \
    --ratio-pca-results outputs/dimensionality_reduction/ratio_pca/tier_based/k10/ \
    --output outputs/evaluations/comparative_analysis/pca_method_comparison/cpca_vs_ratio_k10.json
```

**Compare regularization:**

```bash
python -m probes.scripts.evaluation.compare_regularization_methods \
    --probe-dirs outputs/probes/emotion_probes/text_based/{raw,regularization/l1} \
    --output outputs/evaluations/comparative_analysis/regularization_comparison/l1_vs_none.json
```

## Metrics

### Accuracy Metrics

- **Overall accuracy**: Percentage of correct predictions across all emotions
- **Per-emotion accuracy**: Accuracy for each emotion class
- **User vs Assistant**: Separate metrics for user and assistant turns
- **Macro F1**: Average F1 score across all emotion classes

### Confidence Scores

Softmax probabilities for each emotion prediction:

```python
{
    "happiness": 0.92,
    "sadness": 0.03,
    "anger": 0.02,
    "fear": 0.01,
    "disgust": 0.01,
    "surprise": 0.01
}
```

### Confusion Matrix

Per-emotion confusion showing common misclassifications.

## Conversation Evaluation

### Per-Turn Accuracy

Evaluates each conversation turn independently:
- User turn: First message emotion
- Assistant turn: Response emotion

### Temporal Analysis

How probe accuracy changes across conversation depth:
- Turn 1 accuracy
- Turn 2 accuracy
- Later turns (if available)

### Context Effects

Comparing probe predictions with/without conversation context.

## Emo Lens Experiments

### Single-Layer Experiments

Evaluate one layer at a time to understand layer-wise emotion representation.

**Output includes:**
- `results.json`: Detailed predictions (10-50 KB)
- `visualizations/heatmaps/`: Emotion activation heatmaps
- `visualizations/trajectories/`: Token-level emotion trajectories

### Multilayer Experiments

Comprehensive analysis across all layers.

**Output includes:**
- `results.json`: Very large file (500KB - 1MB)
- `heatmaps/`: Layer × emotion heatmaps showing evolution

**Use cases:**
- Understanding emotion encoding across depth
- Identifying optimal layers for probing
- Analyzing emotion trajectory during generation

## Visualization

Generate plots from evaluation results:

**Accuracy heatmaps:**

```bash
python -m probes.scripts.visualization.visualize_conversation_eval \
    --eval-results outputs/evaluations/conversation_eval/text_based_probes/cpca_variants/*/eval_results*.json \
    --output-dir outputs/evaluations/conversation_eval/plots/accuracy_heatmaps
```

**Multi-seed comparison:**

```bash
python -m probes.scripts.visualization.visualize_multiseed_results \
    --eval-results outputs/evaluations/conversation_eval/text_based_probes/multiseed/*.json \
    --output-dir outputs/visualizations/probe_performance/dimensionality_analysis
```

**Orthogonal probe analysis:**

```bash
python -m probes.scripts.visualization.plot_orthogonal_probe_results \
    --eval-results outputs/evaluations/conversation_eval/conversation_based_probes/*.json \
    --output-dir outputs/visualizations/conversation_analysis/orthogonal_probes
```

## Results Interpretation

### High Accuracy (>95%)
- Probe successfully captures emotion representations
- Representations are linearly separable
- Good for steering and interpretation

### Medium Accuracy (85-95%)
- Decent emotion signal but with noise
- May need more data or better representations
- Consider trying different layers

### Low Accuracy (<85%)
- Poor emotion representation at this layer
- Try different layer or representation method
- May indicate data quality issues

## Common Issues

### Missing Predictions
- Check that conversation format matches expected structure
- Verify probe was trained on compatible emotion set

### Low Confidence Scores
- May indicate ambiguous emotions in data
- Consider reviewing ground truth labels
- Check for distribution shift between train and eval data

### Inconsistent Results
- Run multi-seed evaluation to assess variance
- Check for data leakage between train/test sets
- Verify activation extraction is consistent

## SLURM Scripts

For large-scale evaluations:

```bash
# Evaluate all probe variants
sbatch probes/scripts/slurm_jobs/evaluation/eval_probes.sh

# Evaluate layer 50 (supplementary)
sbatch probes/scripts/slurm_jobs/evaluation/eval_layer50_supplement.sh
```

## Storage

- **Evaluation JSON**: 1-2.5 MB per probe variant
- **Keep only final evaluation results**
- **Archive intermediate/debug runs**
- **Compress old evaluations** (gzip reduces size by ~80%)

## Related Documentation

- [Probe Training](../probes/README.md) - How probes are trained
- [Visualizations](../visualizations/README.md) - Plotting evaluation results
- [Interpretations](../interpretations/README.md) - Understanding probe predictions

---

Last updated: 2025-12-27
