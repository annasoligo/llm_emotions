#!/bin/bash
#SBATCH --job-name=suppress_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/suppress_gemma27b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/suppress_gemma27b_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "Extracting suppression vectors for Gemma 27B"
echo "============================================="

python steering_tests/suppression_experiments/extract_suppression_vectors.py \
    --model gemma3_27b \
    --n_samples 100 \
    --seed 42 \
    --output_dir steering_tests/suppression_experiments/vectors

echo "Done!"
