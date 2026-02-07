#!/bin/bash
#SBATCH --job-name=kl_hf_%a
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --array=0-64

# KL divergence with HuggingFace - all vector types × all layers
# Array index encodes: layer * 5 + vec_type
# Layers: 0,5,10,15,20,25,30,35,40,45,50,55,60 (13 layers)
# Vec types: 0-4 (5 types)
# Total: 13 * 5 = 65 jobs (indices 0-64)

set -e

# Decode array index
LAYERS=(0 5 10 15 20 25 30 35 40 45 50 55 60)
VEC_TYPES=(
    "text_pairs_emotion_vs_neutral"
    "text_pairs_emotion_vs_opposite"
    "text_pairs_emotion_vs_others"
    "base_emotion_vs_others"
    "high_emotion_vs_others"
)

LAYER_IDX=$((SLURM_ARRAY_TASK_ID / 5))
VEC_IDX=$((SLURM_ARRAY_TASK_ID % 5))

LAYER=${LAYERS[$LAYER_IDX]}
VEC_TYPE=${VEC_TYPES[$VEC_IDX]}

echo "=== KL Divergence HF: Layer ${LAYER}, ${VEC_TYPE} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Layer index: $LAYER_IDX, Vec index: $VEC_IDX"
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
    --vector-dir steering_tests/vectors/gemma3_27b/${VEC_TYPE} \
    --num-samples 50 \
    --num-random 5 \
    --layer-norm-pct 1.0 \
    --scales 1 5 10 20 50 100 150

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Total runtime: ${ELAPSED} seconds"
echo "Job completed at $(date)"
