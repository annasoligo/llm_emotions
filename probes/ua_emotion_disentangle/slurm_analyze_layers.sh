#!/bin/bash
#SBATCH --job-name=ua_layer_analysis
#SBATCH --output=/workspace-vast/annas/logs/ua_layer_analysis_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_layer_analysis_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=general

echo "=================================================="
echo "UA Emotion Layer Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date -u)"
echo "=================================================="

cd /workspace-vast/annas/git/research-tools

# Activate environment
source .venv/bin/activate

echo ""
echo "=================================================="
echo "STEP 1: Computing orthogonalized probes"
echo "=================================================="
python3 probes/ua_emotion_disentangle/compute_orthogonal_probes_all_layers.py

echo ""
echo "=================================================="
echo "STEP 2: Analyzing PCs vs dimensions"
echo "=================================================="
python3 probes/ua_emotion_disentangle/analyze_layers_pcs_vs_dimensions.py

echo ""
echo "=================================================="
echo "✓ Analysis Complete"
echo "Finished: $(date -u)"
echo "=================================================="
