#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=reason_L30-40
#SBATCH --output=/workspace-vast/annas/logs/reasoning_evals_layers30-40_%j.out
#SBATCH --error=/workspace-vast/annas/logs/reasoning_evals_layers30-40_%j.err
#SBATCH --time=08:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

MODEL="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-40-2ep-merged"
MODEL_NAME="gemma3_27b_dpo_r64_layers30-40_2ep"

echo "=========================================="
echo "Reasoning Evals: layers30-40"
echo "=========================================="
echo "Model: $MODEL"
echo "Tasks: truthfulqa_mc1, bbh_cot_fewshot, arc_challenge"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

echo ""
echo "=== Running TruthfulQA MC1 ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks truthfulqa_mc1 \
    --batch_size auto \
    --output_path "elicitation/outputs/truthfulqa_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Running BBH CoT Fewshot ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks bbh_cot_fewshot \
    --batch_size auto \
    --output_path "elicitation/outputs/bbh_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Running ARC Challenge ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks arc_challenge \
    --batch_size auto \
    --output_path "elicitation/outputs/arc_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=========================================="
echo "All reasoning evals complete for $MODEL_NAME"
echo "=========================================="
