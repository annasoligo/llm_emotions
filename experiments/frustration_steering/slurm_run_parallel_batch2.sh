#!/bin/bash
#SBATCH --job-name=frustration_steering_b2
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/condition_%a_%j_batch2.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/condition_%a_%j_batch2.err
#SBATCH --array=0-6
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Run 7 conditions in parallel using SLURM array jobs - BATCH 2
# Each array task runs one condition with 20 samples

# Condition mapping
CONDITIONS=(baseline capping ablation steer_anger steer_fear steer_sadness steer_happiness)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Condition: $CONDITION"
echo "Batch: 2 (20 samples)"
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

# Generate responses for this condition - BATCH 2 with 20 samples
echo "Generating responses for condition: $CONDITION (batch 2, 20 samples)"
python generate_multiturn_single_condition.py $CONDITION --num-samples 20 --batch-id 2

if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed for $CONDITION"
    exit 1
fi

echo "✓ Generation complete for $CONDITION (batch 2)"
