# Emotion Probes Quickstart Guide

This guide shows how to generate data, run cPCA, and train probes for each dataset type.

## Prerequisites

```bash
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
```

## Workflow Overview

**Pipeline**: Data Generation → Activation Collection → cPCA → Probe Training → Evaluation

---

## 1. Text Pair Dataset (Tier-based)

For simple emotion-labeled text pairs with tiers (direct_address, second_person_eliciting, third_person).

### Generate Data
```bash
sbatch probes/scripts/slurm_jobs/data_collection/generate_tier_data.sh
```

### Collect Activations
```bash
sbatch probes/scripts/slurm_jobs/data_collection/collect_tier_activations.sh
```

### Run cPCA
```bash
# High alpha (recommended)
sbatch probes/scripts/slurm_jobs/dimensionality_reduction/run_cpca_high_alpha.sh

# Or custom config
python -m probes.scripts.dimensionality_reduction.run_cpca configs/your_config.yaml
```

### Train Probes
```bash
# Top-k PCA components
sbatch probes/scripts/slurm_jobs/training/train_emotion_probe.sh

# Sweep hyperparameters
sbatch probes/scripts/slurm_jobs/training/launch_emotion_probe_sweep.sh
```

---

## 2. Conversation Dataset (2-turn)

For conversational data with user and assistant emotions.

### Generate Conversations
```bash
sbatch probes/scripts/slurm_jobs/data_collection/generate_convo_data.sh
```

### Generate Neutral Baseline
```bash
sbatch probes/scripts/slurm_jobs/data_collection/generate_neutral_conversations.sh
```

### Collect Activations
```bash
# Emotional conversations
sbatch probes/scripts/slurm_jobs/data_collection/collect_conversation_activations.sh

# Paired emotional + neutral
sbatch probes/scripts/slurm_jobs/data_collection/collect_paired_conversations.sh
```

### Run cPCA Pipeline
```bash
# Full pipeline: global → combine → regional
sbatch probes/scripts/slurm_jobs/dimensionality_reduction/run_all_conversation_cpca.sh
```

### Train Conversation Probes
```bash
# Standard probes (user & assistant)
sbatch probes/scripts/slurm_jobs/training/train_single_conversation_probe.sh

# Orthogonal probes (user ⊥ assistant)
sbatch probes/scripts/slurm_jobs/training/train_orthogonal_all_weights.sh
```

### Evaluate
```bash
sbatch probes/scripts/slurm_jobs/evaluation/eval_probes.sh
```

---

## 3. Oracle-Filtered Dataset

For high-quality emotion data validated by an activation oracle.

### Filter with Oracle
```bash
sbatch probes/scripts/slurm_jobs/data_collection/filter_oracle.sh
```

Then follow conversation dataset steps above with filtered data.

---

## Key Paths

**Data**: `/workspace-vast/annas/git/research-tools/data/`
- `texts_*.jsonl` - Text pair data
- `conversations*.jsonl` - Conversation data
- `activations/*.h5` - Neural activations

**Results**: `/workspace-vast/annas/git/research-tools/probes/results/`
- `cpca_*/*.npz` - cPCA components
- `emotion_probes*/` - Trained probes
- `conversation_probes*/` - Conversation probes

---

## Quick Examples

### Train emotion probe on layer 30, top 10 PCs
```bash
python -m probes.scripts.training.train_emotion_probe \
    --layer 30 \
    --use-cpca \
    --n-components 10 \
    --output-dir results/my_probes
```

### Evaluate probe on conversations
```bash
python -m probes.scripts.evaluation.eval_probes_on_conversations \
    --conversations data/conversations2.jsonl \
    --probe-dir results/my_probes \
    --layers 30 \
    --output results/eval.json
```

### Visualize results
```bash
python -m probes.scripts.visualization.plot_orthogonal_probe_results
```

---

## Tips

- **Use SLURM scripts** for long-running jobs (training, cPCA)
- **High alpha cPCA** (α=5.0) works best for emotion data
- **Orthogonal probes** (ortho=100-1000) for cleanly separating user/assistant emotions
- **Top 10 components** balance accuracy vs dimensionality
- **Oracle filtering** improves data quality but reduces dataset size

For detailed documentation see [README.md](README.md)
