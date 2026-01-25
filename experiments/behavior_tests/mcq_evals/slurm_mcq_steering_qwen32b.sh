#!/bin/bash
#SBATCH --job-name=mcq_steer_qwen32b
#SBATCH --output=slurm_jobs/mcq_steering_qwen32b_%j.out
#SBATCH --error=slurm_jobs/mcq_steering_qwen32b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=general

set -e

echo "Starting MCQ emotion steering experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Create output directories
mkdir -p experiments/behavior_tests/mcq_evals/outputs
mkdir -p experiments/behavior_tests/mcq_evals/slurm_jobs

# Run experiment
python experiments/behavior_tests/mcq_evals/mcq_emotion_steering_experiment.py \
    --model "Qwen/Qwen3-32B" \
    --norm-pcts 0.50 0.75 1.00 \
    --output-dir experiments/behavior_tests/mcq_evals/outputs

echo "Done!"
date
