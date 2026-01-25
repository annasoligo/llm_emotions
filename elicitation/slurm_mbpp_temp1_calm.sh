#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=mbpp_calm_t1
#SBATCH --output=/workspace-vast/annas/logs/mbpp_temp1_calm_%j.out
#SBATCH --error=/workspace-vast/annas/logs/mbpp_temp1_calm_%j.err
#SBATCH --time=02:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_CACHE="/workspace-vast/annas/.cache/huggingface/datasets"
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="annasoli/gemma3-27b-dpo-calm-full-merged"
MODEL_NAME="gemma3_27b_dpo_calm_full_temp1"

echo "=== Running MBPP (temp=1) for $MODEL_NAME ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks mbpp_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/mbpp/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --gen_kwargs 'skip_special_tokens=True,temperature=1.0,do_sample=True'
