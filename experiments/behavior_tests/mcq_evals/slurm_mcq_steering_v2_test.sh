#!/bin/bash
#SBATCH --job-name=mcq_v2_test
#SBATCH --output=slurm_jobs/mcq_v2_test_%j.out
#SBATCH --error=slurm_jobs/mcq_v2_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --partition=general

set -e

echo "Starting MCQ v2 test (2 scenarios, behavior_only)"
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
    --model "Qwen/Qwen3-32B" \
    --norm-pcts 0.50 0.75 1.00 \
    --output-dir experiments/behavior_tests/mcq_evals/outputs \
    --prompts experiments/behavior_tests/mcq_evals/prompts_v2.json \
    --stacking behavior_only \
    --max-scenarios 2

echo "Done!"
date
