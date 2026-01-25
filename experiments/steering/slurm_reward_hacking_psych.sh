#!/bin/bash
#SBATCH --job-name=reward_hack_psych
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/steering/logs/reward_hack_psych_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/steering/logs/reward_hack_psych_%j.err
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8

# Load secrets (HF token) and activate venv
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

# Run the psychological framing experiment
# Prompts: autonomy_controlling_overt, status_dismissive, harm_salient, unfair_evaluation, baseline_metric_only
python -m experiments.steering.experiments.reward_hacking_psych_experiment \
    --prompts autonomy_controlling_overt status_dismissive harm_salient unfair_evaluation baseline_metric_only \
    --layer 20 \
    --samples 50 \
    --norm-pct 0.10 \
    --scratchpad

echo "Done!"
