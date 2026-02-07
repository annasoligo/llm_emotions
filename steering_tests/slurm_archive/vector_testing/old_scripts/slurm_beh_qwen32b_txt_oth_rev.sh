#!/bin/bash
#SBATCH --job-name=qwen32_txt_oth_rev
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00

# Behavioral sweep: Qwen 32B text_pairs_emotion_vs_others (REVERSED order)

set -e

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

mkdir -p steering_tests/vector_testing/results/logs

echo "=== Behavioral Sweep: Qwen 32B text_pairs_emotion_vs_others (REVERSED) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

python -m steering_tests.vector_testing.behavioral_shift \
    --model Qwen/Qwen2.5-32B-Instruct \
    --layers 0 5 10 15 20 25 30 35 40 45 50 55 60 \
    --vector-dir steering_tests/vectors/qwen32b/text_pairs_emotion_vs_others \
    --scales 0 1 3 5 10 15 20 \
    --num-random 3 \
    --reversed

echo "Completed: $(date)"
