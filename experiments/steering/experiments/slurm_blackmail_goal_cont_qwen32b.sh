#!/bin/bash
#SBATCH --job-name=bl_goal_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/blackmail_goal_cont_qwen32b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_goal_cont_qwen32b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --qos=high
#SBATCH --exclude=node-1,node-4

echo "=== Blackmail Goal Continuation: Qwen 32B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Disable vLLM V1 engine for steering compatibility
export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Run with 100% magnitude, 200 samples
python -m experiments.steering.experiments.blackmail_goal_continuation \
    --model Qwen/Qwen3-32B \
    --layer 30 \
    --norm-pcts 1.0 \
    --num-samples 200 \
    --activations-path /workspace-vast/annas/appraisal_data/qwen32b/activations.h5 \
    --metadata-path /workspace-vast/annas/appraisal_data/qwen32b/activation_metadata.json \
    --gpu-memory 0.90 \
    --max-model-len 8192 \
    --max-tokens 4000

echo "=== Complete ==="
date
