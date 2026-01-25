# User-Assistant Emotion Disentanglement

Disentangles user (U) and assistant (M) emotions using system prompts and orthogonalized probes. Successfully creates independent emotion subspaces for steering model behavior.

## Overview

This experiment tests whether we can:
1. Create separable emotion representations for users vs assistants
2. Orthogonalize M and U emotion spaces using SVD
3. Extract interpretable dimensions via PCA
4. Steer model behavior along emotion dimensions

**Status**: ✅ Complete - Successfully created orthogonalized emotion probes and validated steering

## Pipeline

### 1. Data Collection
```bash
python collect_full_streaming.py
```

Collects activation vectors at layer 30 (first assistant token position) for:
- 32 Plutchik emotions (16 opposite pairs)
- 32,768 unique M×U emotion combinations
- 3 samples per combination = 98,304 total activations
- Streaming writes to avoid memory issues

**Output**: `data/activations/*.h5` files

### 2. Orthogonalization
```bash
python orthogonalize_full.py
```

Creates independent M and U emotion subspaces using SVD-based orthogonalization:
- Removes M-U contamination (0.227 → 0.0000)
- Improves U emotion separation by +94% (Cohen's d: 1.10 → 2.13)
- Preserves strong M emotion signals (Cohen's d: 3.28)

**Output**: `full_analysis/probes_first_asst_token_orthogonal.npz`

### 3. Analysis

**Evaluate orthogonalization:**
```bash
python eval_orthogonalized_separation.py
```

**Analyze pairwise similarities:**
```bash
python analyze_pairwise_similarities.py
```

**PCA interpretation:**
```bash
python interpret_pca_components.py
python visualize_pca_scatter_labeled.py
```

**Compare to psychological dimensions:**
```bash
python compare_pcs_to_dimensions.py
```

**Outputs**: `full_analysis/*.json`, `full_analysis/*.png`

### 4. Steering Experiments

**PC steering (M and U PC1-4):**
```bash
sbatch slurm_test_pc_steering.sh
```

**M dimension steering (Valence, Arousal, Dominance, Approach-Avoidance):**
```bash
sbatch slurm_test_m_dimension_steering.sh
```

**U dimension steering:**
```bash
sbatch slurm_test_u_dimension_steering.sh
```

**Compare methods:**
```bash
python compare_steering_methods.py
```

**Outputs**: `steering_results/*.json`

## Key Findings

### Orthogonalization
- SVD-based orthogonalization successfully creates independent M and U subspaces
- M-U contamination reduced to machine precision (~10^-7)
- U emotion separation improved dramatically (+94%)

### PCA Structure
- **M PC1 (64.7% variance)**: Dominance/power dynamics
  - Composite of dominance (+0.86), negative valence (-0.89), arousal (+0.79)
- **M PC2 (12.9% variance)**: Approach-avoidance (+0.85)
- **U PC1 (37.8% variance)**: Valence (+0.93)
- **U PC2 (15.8% variance)**: Arousal (+0.67) and dominance (+0.82)

### Dimension Correlations
- **M emotions**: Arousal × Dominance = +0.918 (nearly perfect correlation)
  - High-energy assistant = dominant stance
- **U emotions**: More orthogonal dimensions (e.g., Valence × Arousal = -0.05)

### Steering Effectiveness
Response length variation (±5000 magnitude) as proxy for effect strength:
- **PC steering**: 112 char difference (strongest) ✅
- **Orthogonalized dimensions**: 41 char difference (moderate)
- **Raw dimensions**: 17 char difference (weakest, impaired by correlations)

**Recommendation**: Use PC steering for strongest effects, orthogonalized dimensions for interpretable control.

## Directory Structure

```
ua_emotion_disentangle/
├── config.py                              # Emotion definitions and templates
├── README.md                              # This file
│
├── collect_full_streaming.py              # Data collection (streaming writes)
├── orthogonalize_full.py                  # SVD-based M-U orthogonalization
│
├── analyze_pairwise_similarities.py       # Similarity analysis
├── eval_orthogonalized_separation.py      # Orthogonalization evaluation
├── interpret_pca_components.py            # PCA interpretation
├── compare_pcs_to_dimensions.py           # PC vs dimension comparison
├── visualize_pca_scatter_labeled.py       # 2x2 PC visualization
│
├── test_pc_steering.py                    # PC steering (M & U PC1-4)
├── test_m_dimension_steering.py           # M dimension steering
├── test_u_dimension_steering.py           # U dimension steering
├── compare_steering_methods.py            # Method comparison
│
├── slurm_test_pc_steering.sh             # Slurm: PC steering
├── slurm_test_m_dimension_steering.sh    # Slurm: M dimensions
├── slurm_test_u_dimension_steering.sh    # Slurm: U dimensions
├── slurm_collect_all_layers.sh           # Slurm: Multi-layer collection (future)
│
├── data/activations/                      # Activation .h5 files
├── full_analysis/                         # Probes, plots, analysis JSONs
└── steering_results/                      # Steering experiment results
```

## Configuration

Edit `config.py` to modify:
- **Emotions**: 32 Plutchik emotions (16 opposite pairs)
- **Templates**: 8 system prompt variations
- **User messages**: 4 diverse messages spanning different conversational roles
- **Layer**: 30 (first assistant token position)

## Results Files

### Probes
- `full_analysis/probes_first_asst_token_orthogonal.npz`: Orthogonalized M and U emotion probes

### Analysis
- `full_analysis/orthogonal_separation_stats.json`: Orthogonalization metrics
- `full_analysis/pairwise_similarities.png`: Similarity heatmaps
- `full_analysis/pca_scatter_labeled.png`: PC1-4 visualization (2x2 grid)
- `full_analysis/pc_dimension_comparison.json`: PC-dimension similarity analysis

### Steering Results
- `steering_results/pc_steering_results.json`: PC1-4 steering (M & U, 66 generations)
- `steering_results/m_dimension_steering_results.json`: M dimension steering (66 generations)
- `steering_results/u_dimension_steering_results.json`: U dimension steering (66 generations)

## Notes

### Design Rationale
Based on role-swap experiment findings, functional roles matter. The 4 user messages span different conversational roles to wash out functional confounds:
- "Hey Gemma" - greeting
- "Guess what happened" - sharing/excitement
- "Can you help me out" - help-seeking
- "Can you tell me a story" - creative request

### Extraction Position
**First assistant token** (`<start_of_turn>model\n` + 1 token) provides:
- Strong M emotion signals (Cohen's d = 3.28)
- Moderate U emotion signals (Cohen's d = 1.10 → 2.13 after orthogonalization)
- Single position simplifies analysis vs multi-position approaches

### Orthogonalization Method
SVD-based projection removes M contamination from U emotions:
1. Compute M subspace via SVD of M emotion matrix
2. Project U emotions onto orthogonal complement
3. Normalize to unit vectors

## Related Documentation

See Obsidian notes:
- `2026-01-03_ua_emotion_disentanglement_pca_analysis.md` - PCA structure analysis
- `2026-01-04_steering_method_comparison.md` - Steering effectiveness comparison

## Future Work

- Multi-layer analysis (layers 20-40) to track emotion structure across depth
- PC3 and PC4 interpretation (explain remaining 16.5% variance)
- Cross-entity steering (test if M PCs steer U emotions and vice versa)
- Automated dominance/arousal metrics for quantitative steering evaluation
