#!/bin/bash
#SBATCH --job-name=priority_gemma
#SBATCH --output=/workspace-vast/annas/logs/priority_gemma_%j.out
#SBATCH --error=/workspace-vast/annas/logs/priority_gemma_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00

# Priority selection steering experiment - Gemma
# 20 stimuli × 10 seeds × 25 conditions = 5000 samples

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Priority Selection Steering ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m experiments.behavior_tests.priority_selection_steering \
    --model gemma \
    --emotions fear anger sadness happiness \
    --norm-pcts 0.05 0.075 0.10 \
    --num-seeds 10

echo "Done!"
date
