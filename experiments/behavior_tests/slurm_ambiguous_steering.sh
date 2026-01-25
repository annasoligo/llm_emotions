#!/bin/bash
#SBATCH --job-name=ambiguous_gemma
#SBATCH --output=/workspace-vast/annas/logs/ambiguous_gemma_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ambiguous_gemma_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# Ambiguous interpretation steering experiment - Gemma
# 9 items x 10 permutations x 25 conditions = 2250 samples
# Layer 30 steering, magnitudes 5%, 7.5%, 10%

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo "=== Ambiguous Interpretation Steering - Gemma ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m experiments.behavior_tests.ambiguous_interpretation_steering \
    --model gemma \
    --emotions fear anger sadness happiness \
    --norm-pcts 0.05 0.075 0.10 \
    --num-permutations 10

echo "Done!"
date
