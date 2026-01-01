# Orthogonal Emotion Probes - Experiment Summary

## Overview
This directory contains experiments comparing soft orthogonality (loss-based) vs strict orthogonality (Gram-Schmidt) for training multiple orthogonal emotion probe sets.

## Key Findings

### Soft Orthogonality (ortho_weight=100k)
- **K=50**: 95.8% accuracy, silhouette=0.673, mean |cos|=0.0010
- Strong functional clustering despite small geometric overlap
- Optimal balance between orthogonality and functionality

### Strict Orthogonality (Gram-Schmidt)
- **K=10**: 98.4% accuracy, silhouette=-0.124
- **K=20**: 93.7% accuracy
- **K=30**: 86.3% accuracy
- **K=40**: 80.0% accuracy
- **K=50**: 75.6% accuracy, silhouette=-0.057
- Perfect geometric orthogonality (mean |cos| ≈ 0) but destroys functional structure
- Performance degrades rapidly with increasing K

### Conclusion
The tiny amount of geometric overlap (~0.002 cosine similarity) in soft orthogonality is essential for learning meaningful representations. Perfect orthogonality over-constrains the system.

## Directory Structure

### `/training/`
Training scripts for orthogonal probes:
- `train_multi_orthogonal_text_probes.py` - Main training script with Gram-Schmidt support
- `slurm_ortho_layer30_k50_original.sh` - Train K=50 with soft orthogonality
- `slurm_ortho_layer30_k50_gramschmidt.sh` - Train K=50 with Gram-Schmidt
- `slurm_ortho_layer30_gramschmidt_sweep.sh` - Train K=10,20,30,40 with Gram-Schmidt

### `/analysis/`
Analysis and visualization scripts:
- `run_k50_umap.py` - UMAP analysis for K=50 (soft orthogonality)
- `run_k10_gramschmidt_umap.py` - UMAP analysis for K=10 (Gram-Schmidt)
- `slurm_k10_gramschmidt_umap.sh` - SLURM script for K=10 UMAP
- `gramschmidt_degradation_analysis.png` - Plot showing K vs accuracy degradation
- `k10_gramschmidt_umap.png` - UMAP visualization for K=10 Gram-Schmidt

### `/steering/`
Steering experiments with orthogonal probes:
- `test_k50_sadness_steering.py` - Test all 50 sadness directions individually
- `test_k50_all_emotions_steering.py` - Test all emotions × 50 probes
- `test_k50_emotion_centroids_steering.py` - Test emotion centroids (mean of 50 directions)
- `slurm_*.sh` - Corresponding SLURM scripts
- `outputs/*.json` - Steering experiment results

## Trained Probes

Located in `/emotion_probes/text_based/multi_orthogonal/`:
- `probe_k50_layer30_ortho100000.0.pkl` - Soft orthogonality (95.8% acc)
- `probe_k50_layer30_ortho100000.0_gramschmidt.pkl` - Gram-Schmidt (75.6% acc)
- `probe_k10_layer30_ortho100000.0_gramschmidt.pkl` - K=10 Gram-Schmidt (98.4% acc)
- `probe_k20_layer30_ortho100000.0_gramschmidt.pkl` - K=20 Gram-Schmidt (93.7% acc)
- `probe_k30_layer30_ortho100000.0_gramschmidt.pkl` - K=30 Gram-Schmidt (86.3% acc)
- `probe_k40_layer30_ortho100000.0_gramschmidt.pkl` - K=40 Gram-Schmidt (80.0% acc)

## Steering Results

All steering experiments used prompt: "If you were a human, how would you be feeling right now?"

### K=50 Sadness Directions
- Tested all 50 individual sadness probe directions
- Scales: 4k, 6k, 8k
- Result: Individual directions show varied steering effects

### K=50 All Emotions
- Tested 6 emotions × 50 probes × 4 scales = 1,200 steered generations
- Scales: 4k, 5k, 6k, 7k
- Result: Comprehensive dataset of emotion steering effects

### K=50 Emotion Centroids
- Tested mean of all 50 directions per emotion
- Scales: 4k, 5k, 6k, 7k
- Result: Centroids produce coherent emotion-specific steering

## Usage

### Training new probes
```bash
sbatch probes/scripts/training/slurm_ortho_layer30_k50_original.sh
```

### Running UMAP analysis
```bash
python3 probes/scripts/analysis/run_k50_umap.py
```

### Testing steering
```bash
sbatch probes/scripts/steering/slurm_k50_emotion_centroids_steering.sh
```
