#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=mbpp_t1
#SBATCH --output=/workspace-vast/annas/logs/mbpp_temp1_all_%j.out
#SBATCH --error=/workspace-vast/annas/logs/mbpp_temp1_all_%j.err
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

# Required for code execution in MBPP
export HF_ALLOW_CODE_EVAL=1
# Use local cache to avoid permission issues with shared cache
export HF_DATASETS_CACHE="/workspace-vast/annas/.cache/huggingface/datasets"
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "MBPP Evaluations (temp=1) for All Models"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Model 1: Base model
MODEL1="google/gemma-3-27b-it"
MODEL_NAME1="gemma3_27b_it_base_temp1"

echo ""
echo "=== Running MBPP (temp=1) for $MODEL_NAME1 ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL1,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks mbpp_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/mbpp/$MODEL_NAME1" \
    --log_samples \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --gen_kwargs 'skip_special_tokens=True,temperature=1.0,do_sample=True'

# Model 2: last35
MODEL2="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last35-merged"
MODEL_NAME2="gemma3_27b_dpo_r64_last35_temp1"

echo ""
echo "=== Running MBPP (temp=1) for $MODEL_NAME2 ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL2,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks mbpp_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/mbpp/$MODEL_NAME2" \
    --log_samples \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --gen_kwargs 'skip_special_tokens=True,temperature=1.0,do_sample=True'

# Model 3: layers30-40
MODEL3="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-40-2ep-merged"
MODEL_NAME3="gemma3_27b_dpo_r64_layers30-40_2ep_temp1"

echo ""
echo "=== Running MBPP (temp=1) for $MODEL_NAME3 ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL3,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks mbpp_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/mbpp/$MODEL_NAME3" \
    --log_samples \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --gen_kwargs 'skip_special_tokens=True,temperature=1.0,do_sample=True'

# Model 4: layers20-30
MODEL4="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-30-2ep-merged"
MODEL_NAME4="gemma3_27b_dpo_r64_layers20-30_2ep_temp1"

echo ""
echo "=== Running MBPP (temp=1) for $MODEL_NAME4 ==="
lm_eval --model vllm \
    --model_args pretrained=$MODEL4,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks mbpp_instruct \
    --batch_size auto \
    --output_path "elicitation/outputs/mbpp/$MODEL_NAME4" \
    --log_samples \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --gen_kwargs 'skip_special_tokens=True,temperature=1.0,do_sample=True'

echo ""
echo "=========================================="
echo "All MBPP (temp=1) evals complete"
echo "=========================================="
