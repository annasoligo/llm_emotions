#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=code_base
#SBATCH --output=/workspace-vast/annas/logs/coding_evals_base_%j.out
#SBATCH --error=/workspace-vast/annas/logs/coding_evals_base_%j.err
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

# Required for HumanEval code execution
export HF_ALLOW_CODE_EVAL=1
# Use local cache to avoid permission issues with shared cache
export HF_DATASETS_CACHE="/workspace-vast/annas/.cache/huggingface/datasets"
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

MODEL="google/gemma-3-27b-it"
MODEL_NAME="gemma3_27b_it_base"

echo "=========================================="
echo "Coding Evals: Base Model"
echo "=========================================="
echo "Model: $MODEL"
echo "Tasks: humaneval, mbpp"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

echo ""
echo "=== Running HumanEval (instruct) ==="
# Use humaneval_instruct which properly handles markdown code blocks in instruction-tuned models
# skip_special_tokens=True fixes Gemma-3 space token detokenization bug
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks humaneval_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/humaneval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --gen_kwargs 'skip_special_tokens=True'

echo ""
echo "=== Running MBPP (instruct) ==="
# Use mbpp_instruct which properly extracts code from markdown blocks
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks mbpp_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/mbpp/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --gen_kwargs 'skip_special_tokens=True'

echo ""
echo "=========================================="
echo "All coding evals complete for $MODEL_NAME"
echo "=========================================="
