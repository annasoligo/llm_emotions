#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=appraisal_full
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_full.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_full.err
#SBATCH --time=4:00:00

# Full appraisal data generation run (CPU only - calls Claude API)
#
# Expected output:
#   - 2 axes × 8 cards = 16 cards
#   - 16 cards × 5 domains × 2 scenarios = 160 scenario pairs
#   - 160 pairs × 2 variants × 3 paraphrases = 960 paraphrased texts
#
# Usage:
#   sbatch slurm_full_run.sh

set -e

echo "=========================================="
echo "APPRAISAL FULL DATA GENERATION"
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
echo "Starting pipeline..."

# Run the full config with unbuffered output
python -u -m probes.scripts.appraisal.run \
    --config probes/scripts/appraisal/configs/full_run.yaml 2>&1

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
