#!/bin/bash
#SBATCH --job-name=priority_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/priority_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/priority_qwen235b_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=6:00:00

# Priority selection steering experiment - Qwen 235B
# 20 stimuli × 10 seeds × 25 conditions = 5000 samples
# Layer 50 steering, magnitudes 100%, 125%, 150%

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo "=== Priority Selection Steering - Qwen 235B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m experiments.behavior_tests.priority_selection_steering \
    --model qwen235b \
    --layer 50 \
    --emotions fear anger sadness happiness \
    --norm-pcts 1.0 1.25 1.5 \
    --num-seeds 10

echo "Done!"
date
