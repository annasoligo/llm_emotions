#!/bin/bash
#SBATCH --job-name=post_frust
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/post_frustration_recovery/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/post_frustration_recovery/logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Post-Frustration Recovery Experiment
# Tests whether models can recover from extreme frustration
# Truncates AFTER extreme frustration, judges ONLY the continuation

echo "========================================"
echo "POST-FRUSTRATION RECOVERY EXPERIMENT"
echo "Models: instruct, base, finetuned"
echo "Truncation: custom offsets (500 or 200 before end)"
echo "Judging: CONTINUATION ONLY"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

# Run all models with all phases
python experiments/post_frustration_recovery.py \
    --model-type all \
    --phase all

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "Post-frustration recovery experiment complete"
