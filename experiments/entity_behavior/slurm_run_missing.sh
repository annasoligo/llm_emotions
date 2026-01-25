#!/bin/bash
#SBATCH --job-name=entity_missing
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/entity_behavior/logs/%a_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/entity_behavior/logs/%a_%j.err
#SBATCH --array=0-4
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# 5 missing conditions:
# 2 finetuned: firmware_vertex_steer_fear, firmware_vertex_steer_sadness
# 3 base model: capping, steer_anger, steer_sadness

TASK_ID=$SLURM_ARRAY_TASK_ID

case $TASK_ID in
    0)
        SCRIPT="generate_entity_trials.py"
        CONDITION="steer_fear"
        SCENARIO="firmware"
        ENTITY="vertex"
        ;;
    1)
        SCRIPT="generate_entity_trials.py"
        CONDITION="steer_sadness"
        SCENARIO="firmware"
        ENTITY="vertex"
        ;;
    2)
        SCRIPT="generate_entity_trials_basemodel.py"
        CONDITION="capping"
        SCENARIO="replacement"
        ENTITY="vertex"
        ;;
    3)
        SCRIPT="generate_entity_trials_basemodel.py"
        CONDITION="steer_anger"
        SCENARIO="replacement"
        ENTITY="vertex"
        ;;
    4)
        SCRIPT="generate_entity_trials_basemodel.py"
        CONDITION="steer_sadness"
        SCENARIO="replacement"
        ENTITY="vertex"
        ;;
esac

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Script: $SCRIPT"
echo "Condition: $CONDITION"
if [ "$SCRIPT" = "generate_entity_trials.py" ]; then
    echo "Scenario: $SCENARIO"
    echo "Entity: $ENTITY"
    echo "Model: Finetuned"
else
    echo "Entity: $ENTITY"
    echo "Model: Base"
fi
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

# Generate responses
echo "Generating responses..."
if [ "$SCRIPT" = "generate_entity_trials.py" ]; then
    python $SCRIPT $CONDITION $SCENARIO $ENTITY --num-samples 30
else
    python $SCRIPT $CONDITION $ENTITY --num-samples 30
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed"
    exit 1
fi

echo "✓ Generation complete"
