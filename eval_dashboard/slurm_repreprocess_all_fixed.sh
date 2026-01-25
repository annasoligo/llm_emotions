#!/bin/bash
#SBATCH --job-name=repreprocess_fix
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/annas/logs/repreprocess_fix_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/repreprocess_fix_%A_%a.err
#SBATCH --array=0-11

# Re-run preprocessing for all models with fixed baseline loading
# 4 models × 3 datasets = 12 jobs

set -e

echo "=========================================="
echo "RE-PREPROCESS WITH FIXED BASELINES"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Activate venv
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools

# Define models (HuggingFace paths)
MODELS=(
    "annasoli/gemma3-27b-dpo-calm-full-merged"
    "annasoli/gemma3-27b-dpo-r64-layers20-25-2ep-merged"
    "annasoli/gemma3-27b-dpo-r64-layers30-35-2ep-merged"
    "annasoli/gemma3-27b-sft-last20-3ep-merged"
)

# Define model short names for output files
MODEL_NAMES=(
    "dpo_calm_full"
    "dpo_L20-25"
    "dpo_L30-35"
    "sft_last20_3ep"
)

# Define datasets
DATASETS=(
    "high_emotion_6plus"
    "mid_emotion_3to5"
    "low_emotion_0to2"
)

# Calculate which model and dataset based on array task ID
# Task 0-2: model 0, datasets 0-2
# Task 3-5: model 1, datasets 0-2
# etc.
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 3))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 3))

MODEL=${MODELS[$MODEL_IDX]}
MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

INPUT_FILE="eval_dashboard/data/${DATASET}.pkl"
OUTPUT_FILE="eval_dashboard/data/${DATASET}_${MODEL_NAME}.pkl"

echo ""
echo "Model: $MODEL"
echo "Model Name: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

python eval_dashboard/repreprocess_with_model.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --model "$MODEL"

echo ""
echo "=========================================="
echo "DONE!"
echo "End time: $(date)"
echo "=========================================="
