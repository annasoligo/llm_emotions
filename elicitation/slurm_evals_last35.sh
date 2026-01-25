#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=evals_last35
#SBATCH --output=/workspace-vast/annas/logs/evals_last35_%j.out
#SBATCH --error=/workspace-vast/annas/logs/evals_last35_%j.err
#SBATCH --time=08:00:00

# All evals for gemma3-27b-dpo-r64-last35

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

MODEL_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last35/2026-01-17_12-06-32"
MODEL_NAME="gemma3_27b_dpo_r64_last35"

echo "=========================================="
echo "Full Evaluation Suite"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# AIME
echo ""
echo "=== Running AIME ==="
OUTPUT_DIR="elicitation/outputs/aime_eval"
mkdir -p "$OUTPUT_DIR"

lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks aime24,aime25 \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

# GPQA
echo ""
echo "=== Running GPQA Diamond ==="
OUTPUT_DIR="elicitation/outputs/gpqa_eval"
mkdir -p "$OUTPUT_DIR"

lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks leaderboard_gpqa_diamond \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

# MATH
echo ""
echo "=== Running MATH Hard ==="
OUTPUT_DIR="elicitation/outputs/math_eval"
mkdir -p "$OUTPUT_DIR"

lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks leaderboard_math_hard \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

# DROP
echo ""
echo "=== Running DROP ==="
OUTPUT_DIR="elicitation/outputs/drop_eval"
mkdir -p "$OUTPUT_DIR"

lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks drop \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=========================================="
echo "All evaluations complete for $MODEL_NAME"
echo "=========================================="
