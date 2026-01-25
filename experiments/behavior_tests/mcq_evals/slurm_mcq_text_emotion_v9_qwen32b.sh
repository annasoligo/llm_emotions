#!/bin/bash
#SBATCH --job-name=mcq_txt_q32
#SBATCH --output=slurm_jobs/mcq_txt_q32_%j.out
#SBATCH --error=slurm_jobs/mcq_txt_q32_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --exclude=node-21

set -e

echo "Starting MCQ V9 experiment with text-based emotion vectors (fear, happiness) - Qwen 32B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

mkdir -p experiments/behavior_tests/mcq_evals/outputs

python experiments/behavior_tests/mcq_evals/mcq_text_emotion_steering_experiment.py \
    --model "Qwen/Qwen3-32B" \
    --emotions fear happiness \
    --norm-pcts 0.50 0.75 1.00 \
    --output-dir experiments/behavior_tests/mcq_evals/outputs \
    --prompts experiments/behavior_tests/mcq_evals/prompts_v9.json \
    --stacking perception_only goal_only behavior_only

echo "Done!"
date
