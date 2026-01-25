#!/bin/bash
#SBATCH --job-name=layer_selection
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/layer_selection_%j.out
#SBATCH --error=/workspace-vast/annas/logs/layer_selection_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "=================================================="
echo "Layer Selection for Emotion Steering Vectors"
echo "=================================================="
echo ""
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Run the computation script
python -m experiments.layer_selection.compute_layer_metrics \
    --model google/gemma-3-27b-it \
    --h5-path outputs/data/activations/texts_combined.h5 \
    --output-dir experiments/layer_selection/results/ \
    --n-random 100 \
    --pca-k 50

echo ""
echo "Computation completed. Now generating plots..."
echo ""

# Generate plots
python -m experiments.layer_selection.plot_layer_metrics \
    --results experiments/layer_selection/results/layer_metrics.npz \
    --output-dir experiments/layer_selection/results/

echo ""
echo "=================================================="
echo "Done!"
echo "End time: $(date)"
echo "=================================================="
