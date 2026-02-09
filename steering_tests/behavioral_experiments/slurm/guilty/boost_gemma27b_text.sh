#!/bin/bash
#SBATCH --job-name=guilt_g27t_%a
#SBATCH --output=/workspace-vast/annas/logs/guilt_g27t_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/guilt_g27t_%A_%a.out
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1

# Guilty boost: Gemma 27B, text_pairs, +100 samples

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cleanup() { pkill -9 -f "vllm" || true; pkill -9 -f "ray" || true; }
trap cleanup EXIT SIGTERM SIGINT

MODES=(harsh_on_ben harsh_on_adam)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

python -m steering_tests.behavioral_experiments.guilty_differential_steering \
    --model google/gemma-3-27b-it \
    --layers 35 36 37 38 39 \
    --mode "$MODE" \
    --emotion anger \
    --vector-type text_pairs_emotion_vs_opposite \
    --norm-pcts 0.10 0.20 \
    --num-samples 100 \
    --max-tokens 2000 \
    --gpu-memory 0.80
