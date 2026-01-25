#!/bin/bash
#SBATCH --job-name=psych_prompted
#SBATCH --output=experiments/behavior_tests/logs/psych_prompted_%j.out
#SBATCH --error=experiments/behavior_tests/logs/psych_prompted_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00

# Run prompted experiment on all scenarios (original + additional)

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

# Create log directory
mkdir -p experiments/behavior_tests/logs

echo "Starting prompted psych experiment on all scenarios..."
echo "Timestamp: $(date)"

python -m experiments.behavior_tests.psych_multi_choice_experiment \
    --models gemma \
    --all-scenarios

echo ""
echo "Complete!"
echo "Timestamp: $(date)"
