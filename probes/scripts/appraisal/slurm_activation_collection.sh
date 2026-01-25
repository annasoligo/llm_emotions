#!/bin/bash
#SBATCH --partition=dev
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=appraisal_activations
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_activations.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_activations.err
#SBATCH --time=4:00:00

# Activation collection for appraisal minimal-pair data
# Runs on GPU with Gemma-2-9b-it model
#
# Expected input:
#   - 298 scenario pairs (596 variants)
#   - 596 paraphrase sets (1788 paraphrased texts)
#   Total: ~2384 activation samples
#
# Usage:
#   sbatch slurm_activation_collection.sh

set -e

echo "=========================================="
echo "APPRAISAL ACTIVATION COLLECTION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo ""

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Ensure unbuffered output
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "Starting activation collection..."

# Run only activation logging stage
python -u -m probes.scripts.appraisal.run \
    --config probes/scripts/appraisal/configs/activation_only.yaml \
    --stage activation_logging 2>&1

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
