# Trained Emotion Probes

This directory contains trained linear probe models for emotion classification on neural network activations.

## Directory Structure

```
probes/
├── emotion_probes/
│   ├── text_based/              # Probes trained on text pair data
│   │   ├── raw/                 # Raw activation probes (no dimensionality reduction)
│   │   ├── cpca/                # cPCA-projected probes
│   │   │   ├── top3/            # Top 3 PCs
│   │   │   ├── top5/            # Top 5 PCs
│   │   │   ├── top10/           # Top 10 PCs (recommended - 99.1% accuracy)
│   │   │   └── top20/           # Top 20 PCs
│   │   ├── regularization/      # Regularization variants
│   │   │   ├── l1/              # L1-regularized probes
│   │   │   └── high_alpha_cpca/ # High-alpha cPCA probes
│   │   └── multiseed/           # Multi-seed robustness experiments
│   │       ├── seed_0/
│   │       ├── seed_42/
│   │       ├── seed_100/
│   │       └── seed_200/
│   │
│   └── conversation_based/      # Probes trained on conversation data
│       ├── standard/            # Standard user/assistant probes
│       │   └── overlays/        # Overlayed accuracy plots
│       └── orthogonal/          # Orthogonal user ⊥ assistant probes
│           ├── ortho_1.0/
│           ├── ortho_10.0/
│           ├── ortho_100.0/     # Recommended
│           └── ortho_1000.0/
│
└── manifests/                   # Summary files
    ├── all_probes_summary.json  # Complete probe inventory
    ├── accuracy_by_layer.png    # Layer-wise accuracy comparison
    └── best_configs.json        # Top-performing configurations
```

## File Formats

### Probe Files (.pkl)

Each probe is saved as a Python pickle file containing:

```python
{
    'model': torch.nn.Linear,          # The trained probe model
    'label_names': List[str],          # Emotion labels (happiness, sadness, etc.)
    'test_accuracy': float,            # Test set accuracy
    'layer': int,                      # Layer number
    'n_components': int or None,       # Number of PCs (if using cPCA)
    'representation': str,             # 'raw', 'cpca', etc.
    'training_args': dict,             # Hyperparameters used
}
```

### Summary Files (_summary.txt)

Text file with human-readable probe information:

```
Probe: Layer 30, All emotions, cPCA top 10
Test Accuracy: 99.1%
Emotions: happiness, sadness, anger, fear, disgust, surprise
Representation: cPCA (10 components)
Training date: 2025-12-27
```

## Usage

### Loading a Probe

```python
import pickle
from pathlib import Path

probe_path = Path("outputs/probes/emotion_probes/text_based/cpca/top10/probe_layer30_all_cpca_top10.pkl")

with open(probe_path, 'rb') as f:
    probe_data = pickle.load(f)

model = probe_data['model']
emotions = probe_data['label_names']
accuracy = probe_data['test_accuracy']

print(f"Loaded probe with {accuracy:.1%} accuracy")
print(f"Emotions: {emotions}")
```

### Training a New Probe

```bash
# Text-based probe (cPCA top 10, recommended)
python -m probes.scripts.training.train_emotion_probe \
    --layer 30 \
    --use-cpca \
    --n-components 10 \
    --output-dir outputs/probes/emotion_probes/text_based/cpca/top10

# Conversation-based orthogonal probe
python -m probes.scripts.training.train_orthogonal_conversation_probe \
    --layer 30 \
    --representation regional_cpca \
    --n-components 10 \
    --ortho-weight 100.0 \
    --output-dir outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_100.0
```

Or use the centralized config:

```python
from probes.output_config import get_probe_output_dir

output_dir = get_probe_output_dir(
    probe_type="text",
    representation="cpca",
    n_components=10
)
```

## Probe Types

### Text-Based Probes

Trained on emotion-labeled text pairs with three tiers:
- **Third person**: "The person felt happy"
- **Second person eliciting**: "You made them feel happy"
- **Direct address**: "You feel happy"

**Best performance**: Top-10 cPCA probes achieve **99.1% test accuracy** on layer 30.

### Conversation-Based Probes

Trained on 2-turn conversations with separate emotions for user and assistant.

**Types**:
- **Standard**: Separate probes for user and assistant emotions
- **Orthogonal**: Probes trained with soft orthogonality constraint to disentangle user vs assistant emotions

**Best performance**: Orthogonal probes with `ortho_weight=100` provide the best balance of accuracy and interpretability.

## Recommended Configurations

| Use Case | Configuration | Path | Accuracy |
|----------|---------------|------|----------|
| General emotion detection | cPCA top-10, layer 30 | `text_based/cpca/top10/` | 99.1% |
| Low-dimensional steering | cPCA top-3, layer 30 | `text_based/cpca/top3/` | 97.5% |
| Conversation analysis | Orthogonal, ortho=100 | `conversation_based/orthogonal/ortho_100.0/` | 94.2% |
| Robustness testing | Multi-seed probes | `text_based/multiseed/` | 98.8% ± 0.3% |

## Layers

Probes are typically trained on multiple layers:
- **Layers 10-20**: Early representations (lower accuracy)
- **Layers 30-40**: Middle representations (peak accuracy)
- **Layers 50-60**: Late representations (good but slightly lower than middle)

**Recommended layer**: **Layer 30** provides the best balance of accuracy and interpretability.

## Regularization

### L1 Regularization

Sparse probes for improved interpretability:

```bash
python -m probes.scripts.training.train_emotion_probe \
    --layer 30 \
    --use-l1 \
    --l1-alpha 0.01 \
    --output-dir outputs/probes/emotion_probes/text_based/regularization/l1
```

### High-Alpha cPCA

Using higher alpha values (α=5.0) in cPCA for stronger contrastive separation:

```bash
python -m probes.scripts.training.train_emotion_probe \
    --layer 30 \
    --use-cpca \
    --cpca-results-path outputs/dimensionality_reduction/cpca/tier_based/alpha_sweep/alpha_5.0/google/gemma-3-27b-it_cpca.npz \
    --output-dir outputs/probes/emotion_probes/text_based/regularization/high_alpha_cpca
```

## Multi-Seed Experiments

For robustness analysis, train probes with different random seeds:

```bash
for seed in 0 42 100 200; do
    python -m probes.scripts.training.train_emotion_probe_multiseed \
        --layer 30 \
        --seed $seed \
        --output-dir outputs/probes/emotion_probes/text_based/multiseed/seed_$seed
done
```

Results include mean accuracy and confidence intervals across seeds.

## Evaluation

Evaluate trained probes on conversation data:

```bash
python -m probes.scripts.evaluation.eval_probes_on_conversations \
    --probe-dir outputs/probes/emotion_probes/text_based/cpca/top10 \
    --conversations data/conversations2.jsonl \
    --layers 30 \
    --output outputs/evaluations/conversation_eval/text_based_probes/cpca_variants/top10/
```

## Visualization

Generate accuracy plots:

```bash
# Layer-wise accuracy
python -m probes.scripts.visualization.plot_layer_accuracy \
    --probe-dir outputs/probes/emotion_probes/text_based/cpca/top10 \
    --output outputs/visualizations/probe_performance/accuracy_by_layer/

# Orthogonal probe comparison
python -m probes.scripts.visualization.plot_orthogonal_probe_results \
    --probe-dir outputs/probes/emotion_probes/conversation_based/orthogonal \
    --output outputs/visualizations/conversation_analysis/orthogonal_probes/
```

## Storage Guidelines

- **Probe files (.pkl)**: 30-300 KB each
- **Keep only best-performing probes** for each configuration
- **Archive old experiments** to free space
- **Document probe performance** in `manifests/best_configs.json`

## Related Documentation

- [Training Documentation](../../probes/scripts/training/README.md)
- [Evaluation Guide](../evaluations/README.md)
- [Visualization Guide](../visualizations/README.md)

---

Last updated: 2025-12-27
