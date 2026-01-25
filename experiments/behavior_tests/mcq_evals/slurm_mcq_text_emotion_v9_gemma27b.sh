#!/bin/bash
#SBATCH --job-name=mcq_text_emo
#SBATCH --output=slurm_jobs/mcq_text_emo_%j.out
#SBATCH --error=slurm_jobs/mcq_text_emo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --qos=high

set -e

echo "Starting MCQ V9 experiment with text-based emotion vectors (fear, happiness)"
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
    --model "unsloth/gemma-3-27b-it" \
    --emotions fear happiness \
    --norm-pcts 0.05 0.07 0.10 \
    --output-dir experiments/behavior_tests/mcq_evals/outputs \
    --prompts experiments/behavior_tests/mcq_evals/prompts_v9.json \
    --stacking perception_only goal_only behavior_only

echo "Done!"
date
