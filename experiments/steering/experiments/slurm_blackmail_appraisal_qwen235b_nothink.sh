#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --job-name=bl_nothink
#SBATCH --output=/workspace-vast/annas/logs/blackmail_appraisal_nothink_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_appraisal_nothink_%j.err
#SBATCH --time=8:00:00

# Blackmail with appraisal axes - THINKING DISABLED
# Qwen3-235B at 100%, 125%, 150% magnitude

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Blackmail Appraisal: THINKING DISABLED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -m experiments.steering.experiments.blackmail_appraisal_scratchpad \
    --model Qwen/Qwen3-235B-A22B \
    --layer 50 \
    --num-samples 100 \
    --norm-pcts 1.0 1.25 1.5 \
    --gpu-memory 0.90 \
    --max-model-len 8192 \
    --activations-path /workspace-vast/annas/appraisal_data/qwen235b/activations.h5 \
    --metadata-path /workspace-vast/annas/appraisal_data/qwen235b/activation_metadata.json \
    --no-thinking

echo "=== Complete ==="
date
