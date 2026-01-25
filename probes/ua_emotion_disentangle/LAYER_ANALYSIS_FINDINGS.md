# Layer-wise Evolution of Emotion Dimensions and Principal Components

**Date**: January 4, 2026
**Data**: 131GB activations from Gemma-3-27B across all 62 layers (32,768 emotion combinations)
**Method**: Alternating orthogonalization of M (Assistant) and U (User) emotion probes

## Overview

We analyzed how emotion representations evolve across model layers by:
1. Computing orthogonalized emotion probes for each layer using alternating orthogonalization
2. Extracting PC1-4 from orthogonalized M and U probes separately
3. Computing psychological dimension directions (Valence, Arousal, Dominance, Approach-Avoidance)
4. Measuring cosine similarities between PCs and dimensions across all 62 layers

## Key Findings

### 1. PC1 Shifts from Arousal to Valence Representation

**Early layers (0-10)**: PC1 primarily captures **Arousal** (emotional intensity/energy)
- Assistant PC1 → Arousal: 0.94 absolute cosine similarity
- User PC1 → Arousal: 0.94 absolute cosine similarity

**Middle-late layers (20-60)**: PC1 pivots to **Valence** (positive-negative affect)
- Assistant PC1 → Valence: 0.87-0.93 similarity
- User PC1 → Valence: 0.87-0.94 similarity

**Interpretation**: The model initially encodes emotional intensity, then develops positive-negative affect representation in deeper layers. This suggests a hierarchical processing where basic intensity is computed first, followed by affective valence.

### 2. Valence-Arousal Orthogonalization

| Layer | M V-A Correlation | U V-A Correlation |
|-------|-------------------|-------------------|
| 0     | +0.41            | +0.75            |
| 10    | -0.07            | +0.10            |
| 20    | -0.35            | -0.29            |
| 30    | -0.34            | -0.56            |
| 40    | -0.35            | -0.36            |
| 50    | -0.31            | -0.38            |
| 60    | -0.26            | -0.36            |

**Early layers**: Valence and Arousal are **positively correlated** - the dimensions are entangled. High arousal tends to co-occur with positive valence in the representation.

**Mid-late layers**: Correlation becomes **negative or near-zero** - the dimensions become orthogonal or anti-correlated.

**Key observation**: User emotions start with stronger V-A entanglement (0.75) than Assistant emotions (0.41), but both converge to similar orthogonal structure by layer 20.

**Interpretation**: The model learns to disentangle valence from arousal, developing independent psychological axes. This matches psychological theory where valence and arousal are orthogonal dimensions in circumplex models of emotion.

### 3. Approach-Avoidance Collapses Into Valence

| Layer | M AA-V Correlation | U AA-V Correlation |
|-------|--------------------|--------------------|
| 0     | 0.42              | 0.43              |
| 10    | 0.83              | 0.85              |
| 20    | 0.83              | 0.87              |
| 30    | 0.88              | 0.88              |
| 40    | 0.81              | 0.84              |
| 50    | 0.81              | 0.77              |
| 60    | 0.81              | 0.73              |

**Early layers**: Approach-Avoidance and Valence are moderately correlated (0.42-0.43).

**Layer 10+**: AA-V correlation **jumps to 0.83-0.88** and remains stable.

**Interpretation**: By mid-layers, approach/avoidance becomes essentially **identical to valence**. This is psychologically plausible - positive emotions drive approach behavior, negative emotions drive avoidance/withdrawal. The model learns this fundamental connection between affect and motivated action.

### 4. Consistent Patterns Across Entities

Assistant (M) and User (U) emotion representations show nearly **identical developmental trajectories**:
- Same PC1 shift from Arousal → Valence
- Same V-A orthogonalization pattern
- Same AA-V convergence
- Similar timescales (major changes by layer 10-20)

**Interpretation**: These are **general principles of emotion representation** in the model, not entity-specific quirks. The model develops a shared emotion processing architecture regardless of whether emotions are attributed to assistant or user.

## Visualizations

### PC Similarities to Dimensions
- **`m_pc_to_dimensions.png`**: 4 subplots (PC1-4) showing Assistant PC alignment to each psychological dimension across layers
- **`u_pc_to_dimensions.png`**: 4 subplots (PC1-4) showing User PC alignment to each psychological dimension across layers

### Dimension Pairwise Similarities
- **`m_dimension_similarities.png`**: 4 subplots showing how each Assistant dimension relates to the other 3 dimensions across layers
- **`u_dimension_similarities.png`**: 4 subplots showing how each User dimension relates to the other 3 dimensions across layers

All plots show smooth transitions from layer 0 → 61, revealing the gradual emergence of psychological structure.

## Technical Details

### Data Collection
- **Model**: `unsloth/gemma-3-27b-it` (62 layers, hidden dim 5376)
- **Emotions**: 32 Plutchik emotions (16 opposite pairs + dyads)
- **Templates**: 8 system prompt templates
- **User messages**: 4 variations
- **Total combinations**: 32,768 (32 M emotions × 32 U emotions × 8 templates × 4 messages)
- **Extraction position**: `first_asst_token` (first token of assistant response)
- **Total data**: 131GB raw activations

### Orthogonalization Method
Used **alternating orthogonalization** with damping factor 0.5:
1. Project U probes away from M subspace
2. Project M probes away from U subspace
3. Iterate until convergence (< 1e-6 change)

This removes M-U contamination before computing PCs and dimensions, ensuring we measure genuine emotion structure rather than entity distinctions.

### Psychological Dimensions
Based on dimensional models of emotion:
- **Valence (V)**: Positive vs negative affect
- **Arousal (A)**: High vs low energy/activation
- **Dominance (D)**: Power/control vs submission
- **Approach-Avoidance (AA)**: Approach vs withdrawal motivation

Each dimension computed as: `mean(high emotions) - mean(low emotions)`, then normalized.

## Implications

### 1. Hierarchical Emotion Processing
The PC1 shift suggests **hierarchical processing**:
- Early layers: Detect emotional intensity (arousal)
- Later layers: Classify emotional valence (good/bad)

This mirrors human emotion processing where intensity detection precedes valence evaluation.

### 2. Emergent Psychological Structure
The V-A orthogonalization shows the model **learns textbook psychology**:
- Circumplex models of emotion posit orthogonal valence-arousal axes
- The model discovers this structure without explicit supervision
- Structure emerges gradually across layers 0-20

### 3. Motivation-Affect Integration
The AA-V convergence reveals the model learns the **fundamental connection** between:
- Affective states (valence)
- Motivated behavior (approach/avoidance)

This integration occurs by layer 10 and remains stable, suggesting it's a core organizing principle.

### 4. Entity-General Representations
Identical M and U patterns indicate:
- Emotion processing is a **shared cognitive module**
- Not specialized per entity (assistant vs user)
- Suggests unified emotion understanding across contexts

## Future Directions

1. **Steering experiments**: Use orthogonalized PCs from specific layers to manipulate emotion dimensions
2. **Cross-layer analysis**: How do early-layer arousal encodings influence late-layer valence?
3. **Dimension interactions**: Beyond pairwise - how do all 4 dimensions jointly structure the space?
4. **Other models**: Do these patterns generalize to Claude, GPT-4, Llama?
5. **Finer-grained dimensions**: Big Five personality traits, discrete emotions, social emotions

## Files

### Outputs
- `layer_analysis/m_pc_to_dimensions.png` (676 KB)
- `layer_analysis/u_pc_to_dimensions.png` (665 KB)
- `layer_analysis/m_dimension_similarities.png` (445 KB)
- `layer_analysis/u_dimension_similarities.png` (415 KB)
- `layer_analysis/layer_analysis_results.json` (159 KB) - Full numerical results

### Code
- `compute_orthogonal_probes_all_layers.py` - Orthogonalization for each layer
- `analyze_layers_pcs_vs_dimensions.py` - PCA and dimension analysis
- `orthogonalize_alternating.py` - Alternating orthogonalization method

### Data
- `data/activations/full_all_layers/` - Raw activations (131GB, 62 layer files)
- `orthogonal_probes_all_layers/` - Orthogonalized probes per layer (62 .npz files)

---

**Generated**: 2026-01-04
**Runtime**: ~3 hours (2:52 for data collection, 0:24 for analysis)
