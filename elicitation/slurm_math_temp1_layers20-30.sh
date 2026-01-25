#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=math_L2030
#SBATCH --output=/workspace-vast/annas/logs/math_temp1_layers20-30_%j.out
#SBATCH --error=/workspace-vast/annas/logs/math_temp1_layers20-30_%j.err
#SBATCH --time=02:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_DATASETS_CACHE="/workspace-vast/annas/.cache/huggingface/datasets"
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-30-2ep-merged"
MODEL_NAME="gemma3_27b_dpo_r64_layers20-30_2ep_temp1"
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks minerva_math \
    --batch_size auto \
    --output_path "elicitation/outputs/math_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --gen_kwargs 'temperature=1.0,do_sample=True'
