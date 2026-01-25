#!/bin/bash
#SBATCH --job-name=mcq_v5_235b
#SBATCH --output=slurm_jobs/mcq_v5_235b_%j.out
#SBATCH --error=slurm_jobs/mcq_v5_235b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --partition=general

set -e

echo "Starting MCQ v5 experiment on Qwen 235B (model-relevant scenarios)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

mkdir -p experiments/behavior_tests/mcq_evals/outputs

python experiments/behavior_tests/mcq_evals/mcq_emotion_steering_experiment.py \
    --model "Qwen/Qwen3-235B-A22B" \
    --norm-pcts 0.50 0.75 1.00 \
    --output-dir experiments/behavior_tests/mcq_evals/outputs \
    --prompts experiments/behavior_tests/mcq_evals/prompts_v5.json \
    --stacking perception_only goal_only behavior_only

echo "Done!"
date
