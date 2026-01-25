#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=appraisal_test
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_test.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_test.err

# Appraisal pipeline smoke test - CPU only (calls Claude API)
#
# Usage:
#   sbatch slurm_test.sh

set -e

echo "=========================================="
echo "APPRAISAL PIPELINE SMOKE TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Load authentication (includes ANTHROPIC_API_KEY)
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

# Run the test config (unbuffered for real-time output)
echo "Resetting pipeline progress..."
python -u -m probes.scripts.appraisal.run \
    --config probes/scripts/appraisal/configs/test_small.yaml \
    --no-resume \
    --reset

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
