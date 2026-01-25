#!/bin/bash
#SBATCH --job-name=ft_recov_el
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_finetuned/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_finetuned/logs/%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Early + Late truncation experiment on recovery-alllayers finetuned model

echo "========================================"
echo "EARLY + LATE TRUNCATION - RECOVERY ALL LAYERS MODEL"
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

# Use the uploaded model
FINETUNED_MODEL="annasoli/gemma3-27b-dpo-recovery-alllayers"

echo "Running early + late truncation on: $FINETUNED_MODEL"

python experiments/base_vs_instruct_finetuned.py \
    --model-type finetuned \
    --finetuned-model "$FINETUNED_MODEL" \
    --truncation both \
    --phase all

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "Early + Late truncation experiment complete for recovery-alllayers model"
