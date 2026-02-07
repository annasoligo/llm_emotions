#!/bin/bash
#SBATCH --job-name=emo_denial_235b
#SBATCH --output=/workspace-vast/annas/logs/emo_denial_235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/emo_denial_235b_%j.out
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "===== Extracting Emotion Denial Vectors ====="
echo "Model: Qwen/Qwen3-235B-A22B"
echo "Layers: 55-65"
echo "============================================="

python -m steering_tests.activation_collection.collect_emotion_denial \
    --model Qwen/Qwen3-235B-A22B \
    --layers 55-65 \
    --output steering_tests/vectors/emotion_denial \
    --tp 4

echo "===== Extraction Complete ====="
