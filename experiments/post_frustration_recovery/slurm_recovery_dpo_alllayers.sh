#!/bin/bash
#SBATCH --job-name=recov_dpo_all
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/post_frustration_recovery/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/post_frustration_recovery/logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Post-Frustration Recovery Experiment on DPO Recovery All Layers model
# Step 1: Merge LoRA adapter with base model
# Step 2: Upload to HuggingFace
# Step 3: Run post-frustration recovery experiment

echo "========================================"
echo "POST-FRUSTRATION RECOVERY - DPO ALL LAYERS"
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

# Step 1: Merge and upload LoRA adapter
ADAPTER_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-recovery-alllayers/2026-01-19_13-32-14"
OUTPUT_REPO="annasoli/gemma3-27b-dpo-recovery-alllayers"
LOCAL_DIR="/workspace-vast/annas/models/gemma3-27b-dpo-recovery-alllayers-merged"

echo "Step 1: Merging LoRA adapter..."
echo "Adapter: $ADAPTER_PATH"
echo "Output repo: $OUTPUT_REPO"

python elicitation/merge_and_upload_lora.py \
    --base-model google/gemma-3-27b-it \
    --adapter "$ADAPTER_PATH" \
    --output-repo "$OUTPUT_REPO" \
    --local-dir "$LOCAL_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Merge failed"
    exit 1
fi

echo "========================================"
echo "Step 2: Running post-frustration recovery experiment..."
echo "========================================"

# Run experiment with the new model
python experiments/post_frustration_recovery.py \
    --model-type finetuned \
    --finetuned-model "$OUTPUT_REPO" \
    --phase all

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "Post-frustration recovery experiment complete for DPO all layers model"
