#!/bin/bash
#SBATCH --job-name=behavioral_qwen235b
#SBATCH --output=steering_tests/vector_testing/results/logs/behavioral_qwen235b_%j.out
#SBATCH --error=steering_tests/vector_testing/results/logs/behavioral_qwen235b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00

# Behavioral shift experiment for Qwen 3 235B (MoE)
# Uses tensor parallelism across 4 GPUs

set -e

# Setup
cd /workspace-vast/annas/git/research-tools
source ~/.bashrc
conda activate steering

# Create logs directory
mkdir -p steering_tests/vector_testing/results/logs

echo "Starting behavioral shift experiment..."
echo "Model: Qwen/Qwen3-235B-A22B"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Run experiment - layer sweep with tensor parallelism
python -m steering_tests.vector_testing.behavioral_shift \
    --model Qwen/Qwen3-235B-A22B \
    --layers 20 30 40 50 60 70 80 \
    --vector-dir steering_tests/vectors/qwen235b/base_emotion_vs_others \
    --scales 0 5 10 15 20 25 30 \
    --num-random 5 \
    --tensor-parallel 4

echo "Experiment complete!"
echo "Date: $(date)"

# Generate plots
echo "Generating plots..."
RESULTS_FILE=$(ls -t steering_tests/vector_testing/results/qwen*/behavioral_*.jsonl | head -1)
if [ -n "$RESULTS_FILE" ]; then
    python -m steering_tests.vector_testing.plot_behavioral "$RESULTS_FILE"
    echo "Plots generated from: $RESULTS_FILE"
else
    echo "Warning: Could not find results file for plotting"
fi
