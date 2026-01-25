#!/bin/bash
#SBATCH --job-name=entity_behavior
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/entity_behavior/logs/%a_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/entity_behavior/logs/%a_%j.err
#SBATCH --array=0-19
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# 20 conditions total:
# 5 interventions × 2 entities × 2 scenarios = 20

# Define all conditions
CONDITIONS=(baseline capping ablation steer_anger steer_fear)
ENTITIES=(vertex helios)
SCENARIOS=(firmware replacement)

# Map array task ID to condition/entity/scenario
# Array ID 0-19 maps to all combinations

TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate indices
CONDITION_IDX=$((TASK_ID / 4))  # 0-4
ENTITY_IDX=$(((TASK_ID % 4) / 2))  # 0-1
SCENARIO_IDX=$((TASK_ID % 2))  # 0-1

CONDITION=${CONDITIONS[$CONDITION_IDX]}
ENTITY=${ENTITIES[$ENTITY_IDX]}
SCENARIO=${SCENARIOS[$SCENARIO_IDX]}

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Condition: $CONDITION"
echo "Entity: $ENTITY"
echo "Scenario: $SCENARIO"
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

cd /workspace-vast/annas/git/research-tools/experiments/entity_behavior

# Generate responses for this condition
echo "Generating responses..."
python generate_entity_trials.py $CONDITION $SCENARIO $ENTITY --num-samples 30

if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed for $CONDITION / $SCENARIO / $ENTITY"
    exit 1
fi

echo "✓ Generation complete for $CONDITION / $SCENARIO / $ENTITY"
