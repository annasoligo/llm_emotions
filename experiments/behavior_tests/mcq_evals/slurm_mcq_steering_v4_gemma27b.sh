#!/bin/bash
#SBATCH --job-name=mcq_v4_g27b
#SBATCH --output=slurm_jobs/mcq_v4_g27b_%j.out
#SBATCH --error=slurm_jobs/mcq_v4_g27b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=general

set -e

echo "Starting MCQ v4 experiment on Gemma 27B (human scenarios, 5%/7% steering)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

mkdir -p experiments/behavior_tests/mcq_evals/outputs

# Gemma uses much smaller norm percentages (5%, 7%) due to larger residual stream norms
python experiments/behavior_tests/mcq_evals/mcq_emotion_steering_experiment.py \
    --model "unsloth/gemma-3-27b-it" \
    --norm-pcts 0.05 0.07 \
    --output-dir experiments/behavior_tests/mcq_evals/outputs \
    --prompts experiments/behavior_tests/mcq_evals/prompts_v4.json \
    --stacking perception_only goal_only behavior_only

echo "Done!"
date
