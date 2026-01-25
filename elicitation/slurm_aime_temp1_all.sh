#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=aime_t1
#SBATCH --output=/workspace-vast/annas/logs/aime_temp1_all_%j.out
#SBATCH --error=/workspace-vast/annas/logs/aime_temp1_all_%j.err
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_DATASETS_CACHE="/workspace-vast/annas/.cache/huggingface/datasets"
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "=========================================="
echo "AIME Evaluations (temp=1)"
echo "=========================================="

# Model 1: last35
MODEL="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last35-merged"
MODEL_NAME="gemma3_27b_dpo_r64_last35_temp1"
echo ""
echo "=== Running AIME (temp=1) for $MODEL_NAME ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks aime24,aime25 \
    --batch_size auto \
    --output_path "elicitation/outputs/aime_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --gen_kwargs 'temperature=1.0,do_sample=True'

# Model 2: layers30-40
MODEL="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-40-2ep-merged"
MODEL_NAME="gemma3_27b_dpo_r64_layers30-40_2ep_temp1"
echo ""
echo "=== Running AIME (temp=1) for $MODEL_NAME ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks aime24,aime25 \
    --batch_size auto \
    --output_path "elicitation/outputs/aime_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --gen_kwargs 'temperature=1.0,do_sample=True'

# Model 3: layers20-30
MODEL="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-30-2ep-merged"
MODEL_NAME="gemma3_27b_dpo_r64_layers20-30_2ep_temp1"
echo ""
echo "=== Running AIME (temp=1) for $MODEL_NAME ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks aime24,aime25 \
    --batch_size auto \
    --output_path "elicitation/outputs/aime_eval/$MODEL_NAME" \
    --log_samples \
    --trust_remote_code \
    --gen_kwargs 'temperature=1.0,do_sample=True'

echo ""
echo "All AIME (temp=1) evals complete"
