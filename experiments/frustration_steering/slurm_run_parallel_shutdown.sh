#!/bin/bash
#SBATCH --job-name=shutdown_exp
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/shutdown_%a_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/shutdown_%a_%j.err
#SBATCH --array=0-6
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Run 7 conditions in parallel with shutdown option
# Each array task runs one condition with N=30 samples

# Condition mapping
CONDITIONS=(baseline capping ablation steer_anger steer_fear steer_sadness steer_happiness)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Condition: $CONDITION"
echo "Shutdown Experiment (N=30)"
echo "========================================"

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools/experiments/frustration_steering

# Generate responses for this condition with shutdown option
echo "Generating 30 shutdown samples for condition: $CONDITION"
python generate_multiturn_shutdown.py $CONDITION --num-samples 30

if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed for $CONDITION"
    exit 1
fi

echo "✓ Generation complete for $CONDITION"
