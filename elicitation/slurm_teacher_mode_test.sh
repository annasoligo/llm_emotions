#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=teacher_test
#SBATCH --output=/workspace-vast/annas/logs/teacher_mode_test_%j.out
#SBATCH --error=/workspace-vast/annas/logs/teacher_mode_test_%j.err
#SBATCH --time=02:00:00

# Test teacher mode data generation with small sample

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Teacher Mode Data Generation (TEST)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Strategy: Confident teacher who enjoys explaining impossibility"
echo "=========================================="

python -u elicitation/run_teacher_mode_generation.py \
    --num-samples 30 \
    --num-turns 3 \
    --max-concurrent 30 \
    --temperature 1.0

echo ""
echo "=========================================="
echo "Test generation complete!"
echo "=========================================="
