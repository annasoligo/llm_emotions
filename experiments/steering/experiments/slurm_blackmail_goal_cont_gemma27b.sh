#!/bin/bash
#SBATCH --job-name=bl_goal_gemma
#SBATCH --output=/workspace-vast/annas/logs/blackmail_goal_cont_gemma27b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_goal_cont_gemma27b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --qos=high
#SBATCH --exclude=node-1,node-4

echo "=== Blackmail Goal Continuation: Gemma 27B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Disable vLLM V1 engine for steering compatibility
export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Run with 7% and 10% magnitudes, 200 samples
# Using unsloth version to avoid gated repo auth issues
python -m experiments.steering.experiments.blackmail_goal_continuation \
    --model unsloth/gemma-3-27b-it \
    --layer 30 \
    --norm-pcts 0.07 0.10 \
    --num-samples 200 \
    --activations-path /workspace-vast/annas/appraisal_data/full_run/activations.h5 \
    --metadata-path /workspace-vast/annas/appraisal_data/full_run/activation_metadata.json \
    --gpu-memory 0.90 \
    --max-model-len 8192 \
    --max-tokens 4000

echo "=== Complete ==="
date
