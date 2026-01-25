#!/bin/bash
#SBATCH --job-name=appraisal_layer_sel
#SBATCH --partition=general
#SBATCH --gpus=0
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/appraisal_layer_selection_%j.out
#SBATCH --error=/workspace-vast/annas/logs/appraisal_layer_selection_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "=================================================="
echo "Appraisal Layer Selection (Valence, Uncertainty, Agency)"
echo "=================================================="
echo ""
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# ============================================
# Gemma 3 27B
# ============================================
echo "=================================================="
echo "Computing metrics for Gemma 3 27B..."
echo "=================================================="

python -m experiments.layer_selection.compute_appraisal_layer_metrics \
    --h5-path /workspace-vast/annas/appraisal_data/full_run/activations.h5 \
    --output-dir experiments/layer_selection/results/appraisal/ \
    --model-name gemma27b \
    --n-random 100 \
    --pca-k 50

echo ""

# ============================================
# Qwen 3 32B
# ============================================
echo "=================================================="
echo "Computing metrics for Qwen 3 32B..."
echo "=================================================="

python -m experiments.layer_selection.compute_appraisal_layer_metrics \
    --h5-path /workspace-vast/annas/appraisal_data/qwen32b/activations.h5 \
    --output-dir experiments/layer_selection/results/appraisal/ \
    --model-name qwen32b \
    --n-random 100 \
    --pca-k 50

echo ""

# ============================================
# Generate Plots
# ============================================
echo "=================================================="
echo "Generating plots..."
echo "=================================================="

python -m experiments.layer_selection.plot_appraisal_layer_metrics \
    --results experiments/layer_selection/results/appraisal/appraisal_layer_metrics_gemma27b.npz \
             experiments/layer_selection/results/appraisal/appraisal_layer_metrics_qwen32b.npz \
    --output-dir experiments/layer_selection/results/appraisal/

echo ""
echo "=================================================="
echo "Done!"
echo "End time: $(date)"
echo "=================================================="
