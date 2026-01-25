#!/bin/bash
#SBATCH --job-name=u_dim_steering
#SBATCH --output=/workspace-vast/annas/logs/u_dim_steering_%j.out
#SBATCH --error=/workspace-vast/annas/logs/u_dim_steering_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

echo "=================================================="
echo "U (User) Dimension Steering Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=================================================="

source /workspace-vast/annas/git/research-tools/.venv/bin/activate
cd /workspace-vast/annas/git/research-tools

python3 probes/ua_emotion_disentangle/test_u_dimension_steering.py

echo "=================================================="
echo "Completed: $(date)"
echo "=================================================="
