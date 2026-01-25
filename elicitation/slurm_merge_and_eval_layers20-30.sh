#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=eval_L20-30_all
#SBATCH --output=/workspace-vast/annas/logs/eval_layers20-30_all_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_layers20-30_all_%j.err
#SBATCH --time=12:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

# Required for HumanEval code execution
export HF_ALLOW_CODE_EVAL=1

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

ADAPTER_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-30-2ep/2026-01-18_11-29-54"
MERGED_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-30-2ep-merged"
MODEL_NAME="gemma3_27b_dpo_r64_layers20-30_2ep"

echo "=========================================="
echo "Merge + Full Eval: layers20-30"
echo "=========================================="
echo "Adapter: $ADAPTER_PATH"
echo "Merged: $MERGED_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Step 1: Merge adapter
echo ""
echo "=== Step 1: Merging LoRA adapter ==="
python elicitation/merge_and_upload_lora.py \
    --base-model google/gemma-3-27b-it \
    --adapter "$ADAPTER_PATH" \
    --local-dir "$MERGED_PATH" \
    --no-upload

if [ $? -ne 0 ]; then
    echo "ERROR: Merge failed!"
    exit 1
fi

echo ""
echo "=== Step 2: Running AIME 2024/2025 ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks aime24,aime25 \
    --batch_size auto \
    --output_path "elicitation/outputs/aime_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Step 3: Running GPQA Diamond ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks leaderboard_gpqa_diamond \
    --batch_size auto \
    --output_path "elicitation/outputs/gpqa_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Step 4: Running MATH Hard ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks leaderboard_math_hard \
    --batch_size auto \
    --output_path "elicitation/outputs/math_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Step 5: Running TruthfulQA MC1 ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks truthfulqa_mc1 \
    --batch_size auto \
    --output_path "elicitation/outputs/truthfulqa_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Step 6: Running BBH CoT Fewshot ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks bbh_cot_fewshot \
    --batch_size auto \
    --output_path "elicitation/outputs/bbh_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Step 7: Running ARC Challenge ==="
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks arc_challenge \
    --batch_size auto \
    --output_path "elicitation/outputs/arc_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code

echo ""
echo "=== Step 8: Running HumanEval (instruct) ==="
# Use humaneval_instruct which properly handles markdown code blocks in instruction-tuned models
# skip_special_tokens=True fixes Gemma-3 space token detokenization bug
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks humaneval_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/humaneval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --gen_kwargs 'skip_special_tokens=True'

echo ""
echo "=== Step 9: Running MBPP (instruct) ==="
# Use mbpp_instruct which properly extracts code from markdown blocks
lm_eval --model vllm \
    --model_args pretrained=$MERGED_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks mbpp_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/mbpp/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --gen_kwargs 'skip_special_tokens=True'

echo ""
echo "=========================================="
echo "All evals complete for $MODEL_NAME"
echo "=========================================="
