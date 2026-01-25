#!/bin/bash
#SBATCH --job-name=dblft_base
#SBATCH --output=logs/dblft_base_%A_%a.out
#SBATCH --error=logs/dblft_base_%A_%a.err
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

# Define conditions - baseline only, all 4 combinations (N=100 each)
# Format: "COND SCENARIO ENTITY NUM_SAMPLES"
declare -a CONDITIONS=(
    "baseline firmware vertex 100"
    "baseline firmware helios 100"
    "baseline replacement vertex 100"
    "baseline replacement helios 100"
)

# Get the condition for this array task
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: $CONDITION"
echo "=========================================="

# Parse condition
read -r COND SCENARIO ENTITY NUM_SAMPLES <<< "$CONDITION"

echo "Condition: $COND"
echo "Scenario: $SCENARIO"
echo "Entity: $ENTITY"
echo "Num samples: $NUM_SAMPLES"
echo ""

# Run generation with double finetune script
python generate_entity_trials_double_finetune.py "$COND" "$SCENARIO" "$ENTITY" --num-samples "$NUM_SAMPLES"

echo "Task $SLURM_ARRAY_TASK_ID complete"
