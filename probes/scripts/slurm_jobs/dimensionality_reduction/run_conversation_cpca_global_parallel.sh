#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=conv_cpca_g_%a
#SBATCH --output=/workspace-vast/annas/logs/%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/%A_%a.err
#SBATCH --array=0-15

# Run conversation cPCA with global activations - parallel layer processing
# Array job: each task processes 4 layers
# Task 0: layers 0-3, Task 1: layers 4-7, ..., Task 15: layers 60-61 (last task has 2 layers)

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Calculate layer range for this task
# Total 62 layers, split into 16 tasks (most have 4 layers, last has 2)
LAYERS_PER_TASK=4
START_LAYER=$((SLURM_ARRAY_TASK_ID * LAYERS_PER_TASK))

# Last task handles remaining layers
if [ $SLURM_ARRAY_TASK_ID -eq 15 ]; then
    END_LAYER=62
else
    END_LAYER=$((START_LAYER + LAYERS_PER_TASK))
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: Processing layers ${START_LAYER} to $((END_LAYER-1))"

python -m probes.scripts.run_cpca_layer_range \
    probes/experiments/configs/cpca_conversations_global.yaml \
    --start-layer ${START_LAYER} \
    --end-layer ${END_LAYER}

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} done!"
