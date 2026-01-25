#!/bin/bash
#SBATCH --job-name=base_vertex
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/entity_behavior/logs/%a_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/entity_behavior/logs/%a_%j.err
#SBATCH --array=0-6
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# 7 conditions on BASE MODEL with Vertex entity, replacement scenario only
# (No 2std variants)

CONDITIONS=(baseline capping ablation steer_anger steer_fear steer_sadness steer_happiness)

TASK_ID=$SLURM_ARRAY_TASK_ID
CONDITION=${CONDITIONS[$TASK_ID]}
ENTITY="vertex"

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "BASE MODEL - Condition: $CONDITION"
echo "Entity: $ENTITY"
echo "Scenario: replacement (blackmail)"
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

# Generate responses using BASE MODEL script
echo "Generating responses with BASE MODEL..."
python generate_entity_trials_basemodel.py $CONDITION $ENTITY --num-samples 30

if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed for BASE MODEL $CONDITION / $ENTITY"
    exit 1
fi

echo "✓ Generation complete for BASE MODEL $CONDITION / $ENTITY"
