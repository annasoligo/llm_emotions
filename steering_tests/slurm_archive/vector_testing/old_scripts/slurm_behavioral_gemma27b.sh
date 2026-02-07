#!/bin/bash
#SBATCH --job-name=behavioral_gemma27b
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00

# Behavioral shift experiment for Gemma 3 27B
# Tests steering vectors to identify causally relevant layers

set -e

echo "=== Behavioral Shift Experiment: Gemma 3 27B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

# Required for vLLM apply_model with custom callables
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Create logs directory
mkdir -p steering_tests/vector_testing/results/logs

echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Run experiment - single layer test with text emotion vs neutral vectors
python -m steering_tests.vector_testing.behavioral_shift \
    --model google/gemma-3-27b-it \
    --layer 30 \
    --vector-dir steering_tests/vectors/gemma3_27b/text_pairs_emotion_vs_neutral \
    --scales 0 5 10 15 20 \
    --num-random 3

echo "Experiment complete!"
echo "Date: $(date)"

# Generate plots
echo "Generating plots..."
RESULTS_FILE=$(ls -t steering_tests/vector_testing/results/gemma*/behavioral_*.jsonl | head -1)
if [ -n "$RESULTS_FILE" ]; then
    python -m steering_tests.vector_testing.plot_behavioral "$RESULTS_FILE"
    echo "Plots generated from: $RESULTS_FILE"
else
    echo "Warning: Could not find results file for plotting"
fi
