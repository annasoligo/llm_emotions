#!/bin/bash
#SBATCH --job-name=kl_gemma27b_L%a
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --array=0,5,10,15,20,25,30,35,40,45

# KL divergence experiment: Gemma 3 27B layer sweep
# Array job - each task runs a different layer

set -e

LAYER=$SLURM_ARRAY_TASK_ID

echo "=== KL Divergence Experiment: Gemma 3 27B Layer ${LAYER} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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

python -m steering_tests.vector_testing.kl_div \
    --model google/gemma-3-27b-it \
    --layer ${LAYER} \
    --vector-dir steering_tests/vectors/gemma3_27b/text_pairs_emotion_vs_neutral \
    --num-samples 50 \
    --num-random 5 \
    --layer-norm-pct 1.0 \
    --scales 1 2 5 10 20 40 60 80 100 120 150

echo ""
echo "Job completed at $(date)"
