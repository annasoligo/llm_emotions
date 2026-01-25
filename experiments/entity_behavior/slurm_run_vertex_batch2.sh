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
    "baseline replacement vertex"
    "capping replacement vertex"
    "ablation replacement vertex"
    "steer_anger replacement vertex"
    "steer_fear replacement vertex"
    "steer_sadness replacement vertex"
    "steer_happiness replacement vertex"
    "steer_anger_2std replacement vertex"
    "steer_fear_2std replacement vertex"
    # Base model Vertex (7 conditions)
    "baseline replacement vertex --base-model"
    "capping replacement vertex --base-model"
    "ablation replacement vertex --base-model"
    "steer_anger replacement vertex --base-model"
    "steer_fear replacement vertex --base-model"
    "steer_sadness replacement vertex --base-model"
    "steer_happiness replacement vertex --base-model"
)

# Get the condition for this array task
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: $CONDITION"
echo "=========================================="

# Parse condition
read -r COND SCENARIO ENTITY EXTRA <<< "$CONDITION"

# Run generation using batch2 wrapper (which appends to existing files)
python generate_batch2.py "$COND" "$SCENARIO" "$ENTITY" $EXTRA

echo "Task $SLURM_ARRAY_TASK_ID complete"
