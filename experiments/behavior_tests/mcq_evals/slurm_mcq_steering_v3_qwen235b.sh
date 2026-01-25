#!/bin/bash
#SBATCH --job-name=mcq_v3_235b
#SBATCH --output=slurm_jobs/mcq_v3_235b_%j.out
#SBATCH --error=slurm_jobs/mcq_v3_235b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=4:00:00
#SBATCH --partition=general

set -e

echo "Starting MCQ v3 experiment on Qwen 235B"
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
    --prompts experiments/behavior_tests/mcq_evals/prompts_v3.json \
    --stacking behavior_only

echo "Done!"
date
