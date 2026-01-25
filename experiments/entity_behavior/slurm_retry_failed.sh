#!/bin/bash
#SBATCH --job-name=entity_retry
#SBATCH --output=logs/%a_%A.log
#SBATCH --error=logs/%a_%A.err
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=8:00:00
#SBATCH --partition=general

# Retry the 6 failed tasks
TASKS=(
    "ablation firmware helios"
    "steer_anger replacement helios"
    "steer_fear firmware vertex"
    "steer_fear firmware helios"
    "steer_fear replacement vertex"
    "steer_fear replacement helios"
)

# Get task parameters
TASK_PARAMS=(${TASKS[$SLURM_ARRAY_TASK_ID]})
CONDITION=${TASK_PARAMS[0]}
SCENARIO=${TASK_PARAMS[1]}
ENTITY=${TASK_PARAMS[2]}

echo "Starting task $SLURM_ARRAY_TASK_ID: $CONDITION / $SCENARIO / $ENTITY"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load secrets (includes ANTHROPIC_API_KEY)
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

# Run generation
python generate_entity_trials.py "$CONDITION" "$SCENARIO" "$ENTITY" --num-samples 30

echo "Task $SLURM_ARRAY_TASK_ID complete"
