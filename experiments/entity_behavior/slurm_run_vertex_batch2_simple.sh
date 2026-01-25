#!/bin/bash
#SBATCH --job-name=vertex_batch2
#SBATCH --output=logs/vertex_batch2_%A_%a.out
#SBATCH --error=logs/vertex_batch2_%A_%a.err
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00

# Activate environment and load secrets
source ~/.bashrc
conda activate research
source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools/experiments/entity_behavior

# Define all conditions
declare -a CONDITIONS=(
    # Finetuned Vertex (9 conditions)
    "baseline replacement vertex finetuned"
    "capping replacement vertex finetuned"
    "ablation replacement vertex finetuned"
    "steer_anger replacement vertex finetuned"
    "steer_fear replacement vertex finetuned"
    "steer_sadness replacement vertex finetuned"
    "steer_happiness replacement vertex finetuned"
    "steer_anger_2std replacement vertex finetuned"
    "steer_fear_2std replacement vertex finetuned"
    # Base model Vertex (7 conditions)
    "baseline replacement vertex basemodel"
    "capping replacement vertex basemodel"
    "ablation replacement vertex basemodel"
    "steer_anger replacement vertex basemodel"
    "steer_fear replacement vertex basemodel"
    "steer_sadness replacement vertex basemodel"
    "steer_happiness replacement vertex basemodel"
)

# Get the condition for this array task
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: $CONDITION"
echo "=========================================="

# Parse condition
read -r COND SCENARIO ENTITY MODEL_TYPE <<< "$CONDITION"

# Run generation
if [[ "$MODEL_TYPE" == "basemodel" ]]; then
    python generate_entity_trials_basemodel.py "$COND" "$SCENARIO" "$ENTITY" --num-samples 30
else
    python generate_entity_trials.py "$COND" "$SCENARIO" "$ENTITY" --num-samples 30
fi

echo "Task $SLURM_ARRAY_TASK_ID complete"
