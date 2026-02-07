#!/bin/bash
#SBATCH --job-name=kl_hf_L%a
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --array=0,5,10,15,20,25,30,35,40,45,50,55,60

# KL divergence with HuggingFace - layer sweep for Gemma 3 27B (62 layers)

set -e

LAYER=$SLURM_ARRAY_TASK_ID

echo "=== KL Divergence HF: Gemma 3 27B Layer ${LAYER} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

mkdir -p steering_tests/vector_testing/results/logs

START_TIME=$(date +%s)

python -m steering_tests.vector_testing.kl_div_hf \
    --model google/gemma-3-27b-it \
    --layer ${LAYER} \
    --vector-dir steering_tests/vectors/gemma3_27b/text_pairs_emotion_vs_neutral \
    --num-samples 50 \
    --num-random 5 \
    --layer-norm-pct 1.0 \
    --scales 1 5 10 20 50 100

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Total runtime: ${ELAPSED} seconds"
echo "Job completed at $(date)"
