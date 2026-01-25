#!/bin/bash
#SBATCH --job-name=mcq_txt_235
#SBATCH --output=slurm_jobs/mcq_txt_235_%j.out
#SBATCH --error=slurm_jobs/mcq_txt_235_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=360G
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --exclude=node-1,node-2,node-9,node-21

set -e

echo "Starting MCQ V9 experiment with text-based emotion vectors (fear, happiness) - Qwen 235B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0
export NCCL_DEBUG=WARN

mkdir -p experiments/behavior_tests/mcq_evals/outputs

python experiments/behavior_tests/mcq_evals/mcq_text_emotion_steering_experiment.py \
    --model "Qwen/Qwen3-235B-A22B" \
    --emotions fear happiness \
    --norm-pcts 0.50 0.75 1.00 \
    --output-dir experiments/behavior_tests/mcq_evals/outputs \
    --prompts experiments/behavior_tests/mcq_evals/prompts_v9.json \
    --stacking perception_only goal_only behavior_only \
    --gpu-memory 0.90

echo "Done!"
date
