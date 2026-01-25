#!/bin/bash
#SBATCH --job-name=psych_full
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/steering/logs/psych_full_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/steering/logs/psych_full_%j.err
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

# Required for steering hooks to work with vLLM V1 engine
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Run full experiment: MC verbatim + natural language with judging
# All 7 scenarios, 100 samples per condition for natural
python -m experiments.steering.experiments.psych_decision_steering \
    --layer 30 \
    --norm-pct 0.10 \
    --num-samples 100

echo "Done!"
