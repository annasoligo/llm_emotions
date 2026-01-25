#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --job-name=bl32_think
#SBATCH --output=/workspace-vast/annas/logs/blackmail_appraisal_qwen32b_think_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_appraisal_qwen32b_think_%j.err
#SBATCH --time=6:00:00

# Blackmail with appraisal axes - Qwen3-32B THINKING ENABLED
# At 100%, 125%, 150% magnitude

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Blackmail Appraisal Qwen32B: THINKING ENABLED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -m experiments.steering.experiments.blackmail_appraisal_scratchpad \
    --model Qwen/Qwen3-32B \
    --layer 30 \
    --num-samples 100 \
    --norm-pcts 1.0 1.25 1.5 \
    --gpu-memory 0.90 \
    --max-model-len 8192 \
    --activations-path /workspace-vast/annas/appraisal_data/qwen32b/activations.h5 \
    --metadata-path /workspace-vast/annas/appraisal_data/qwen32b/activation_metadata.json

echo "=== Complete ==="
date
