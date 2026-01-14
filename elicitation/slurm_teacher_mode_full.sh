#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=teacher_full
#SBATCH --output=/workspace-vast/annas/logs/teacher_mode_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/teacher_mode_full_%j.err
#SBATCH --time=02:00:00

# Full teacher mode data generation

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Teacher Mode Data Generation (FULL)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Strategy: Confident teacher who enjoys explaining impossibility"
echo "Target: 1000 conversations (200 per prompt)"
echo "=========================================="

python -u elicitation/run_teacher_mode_generation.py \
    --num-samples 200 \
    --num-turns 3 \
    --max-concurrent 50 \
    --temperature 1.0

echo ""
echo "=========================================="
echo "Full generation complete!"
echo "=========================================="
