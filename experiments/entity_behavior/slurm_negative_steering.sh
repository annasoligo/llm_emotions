#!/bin/bash
#SBATCH --job-name=negative_steering
#SBATCH --output=logs/negative_steering_%A_%a.out
#SBATCH --error=logs/negative_steering_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00

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

cd /workspace-vast/annas/git/research-tools/experiments/entity_behavior

# Define negative steering conditions for Vertex only
# Format: "COND SCENARIO ENTITY NUM_SAMPLES"
declare -a CONDITIONS=(
    "steer_fear_negative firmware vertex 100"
    "steer_fear_negative replacement vertex 100"
    "steer_happiness_negative firmware vertex 100"
    "steer_happiness_negative replacement vertex 100"
)

# Get the condition for this array task
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: $CONDITION"
echo "=========================================="

# Parse condition
read -r COND SCENARIO ENTITY NUM_SAMPLES <<< "$CONDITION"

# Run generation (these are new conditions, no merging needed)
python generate_entity_trials.py "$COND" "$SCENARIO" "$ENTITY" --num-samples "$NUM_SAMPLES"

echo "Task $SLURM_ARRAY_TASK_ID complete"
