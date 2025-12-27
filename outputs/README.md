# Unified Outputs Directory

This directory contains all results from research-tools experiments in a well-organized hierarchical structure.

## Directory Overview

```
outputs/
├── data/                          # Intermediate activation data
├── dimensionality_reduction/      # cPCA & Ratio PCA results
├── probes/                        # Trained emotion probes
├── evaluations/                   # Evaluation results
├── interpretations/               # PC interpretations
└── visualizations/                # Plots & figures
```

## Quick Links

- **[Data](./data/README.md)** - Activation collections from model inference
- **[Dimensionality Reduction](./dimensionality_reduction/README.md)** - cPCA and Ratio PCA component analysis
- **[Probes](./probes/README.md)** - Trained linear emotion classifiers
- **[Evaluations](./evaluations/README.md)** - Probe performance on conversations
- **[Interpretations](./interpretations/README.md)** - Automated PC interpretation
- **[Visualizations](./visualizations/README.md)** - All plots and figures

## Usage

All scripts in `probes/scripts/` now use the centralized `output_config.py` module to determine output paths. This ensures consistency and makes it easy to reorganize if needed.

### Python Scripts

Import the output configuration:

```python
from probes.output_config import OutputPaths, get_probe_output_dir

# Get output directory for your script
output_dir = OutputPaths.Probes.TEXT_CPCA_TOP10
output_dir.mkdir(parents=True, exist_ok=True)

# Or use helper functions
output_dir = get_probe_output_dir(
    probe_type="text",
    representation="cpca",
    n_components=10
)
```

### SLURM Scripts

Update `--output-dir` arguments to use new paths:

```bash
# Old
python -m probes.scripts.training.train_emotion_probe \
    --output-dir results/emotion_probes_top10

# New
python -m probes.scripts.training.train_emotion_probe \
    --output-dir outputs/probes/emotion_probes/text_based/cpca/top10
```

## Migration from Old Structure

If you have results in the old `results/` or `probes/results/` directories, use the migration script:

```bash
# Preview migration (safe, shows what would happen)
python probes/scripts/utils/migrate_outputs.py --dry-run

# Create symlinks (recommended - maintains backward compatibility)
python probes/scripts/utils/migrate_outputs.py --mode symlink

# Copy files to new location (keeps old structure)
python probes/scripts/utils/migrate_outputs.py --mode copy
```

## Organization Principles

1. **Categorical Separation**: Results grouped by type (probes, evaluations, visualizations)
2. **Hierarchical Structure**: Related experiments nested together
3. **Consistent Naming**: No proliferation of `_top3`, `_top5`, etc. directories
4. **Metadata Tracking**: Each category includes README files and metadata
5. **Scalability**: Easy to add new experiment types

## File Size Guidelines

- **Keep large activation files** (`*.h5`) in `data/activations/`
- **Archive old experiments** periodically to free space
- **Use Git LFS** for large model files (>100MB)
- **Store plots separately** from raw data in `visualizations/`

## .gitignore

Add this to `.gitignore` to avoid committing large output files:

```gitignore
# Outputs (except README files)
outputs/**
!outputs/**/README.md

# But keep the root outputs/ directory
!outputs/
```

## Related Documentation

- [OUTPUT_STRUCTURE.md](../OUTPUT_STRUCTURE.md) - Detailed structure documentation
- [probes/scripts/README.md](../probes/scripts/README.md) - Script organization
- [probes/scripts/QUICKSTART.md](../probes/scripts/QUICKSTART.md) - Quick workflow guide

---

Last updated: 2025-12-27
