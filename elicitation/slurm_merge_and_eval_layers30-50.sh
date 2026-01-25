#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=eval_L30-50
#SBATCH --output=/workspace-vast/annas/logs/merge_eval_layers30-50_%j.out
#SBATCH --error=/workspace-vast/annas/logs/merge_eval_layers30-50_%j.err
#SBATCH --time=10:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

ADAPTER_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-50-2ep/2026-01-17_18-56-24"
MERGED_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-50-2ep-merged"
MODEL_NAME="gemma3_27b_dpo_r64_layers30-50_2ep"

echo "=========================================="
echo "Merge and Evaluate: layers30-50-2ep"
echo "=========================================="
echo "Adapter: $ADAPTER_PATH"
echo "Merged output: $MERGED_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Step 1: Merge adapter
echo ""
echo "=== Merging LoRA adapter ==="
python elicitation/merge_and_upload_lora.py \
    --base-model google/gemma-3-27b-it \
    --adapter "$ADAPTER_PATH" \
    --local-dir "$MERGED_PATH" \
    --no-upload

# Step 2: Run evals
echo ""
echo "=== Running AIME ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks aime24,aime25 \
    --batch_size auto \
    --output_path "elicitation/outputs/aime_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Running GPQA Diamond ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks leaderboard_gpqa_diamond \
    --batch_size auto \
    --output_path "elicitation/outputs/gpqa_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Running MATH Hard ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks leaderboard_math_hard \
    --batch_size auto \
    --output_path "elicitation/outputs/math_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Running DROP ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks drop \
    --batch_size auto \
    --output_path "elicitation/outputs/drop_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=========================================="
echo "All evaluations complete for $MODEL_NAME"
echo "=========================================="
