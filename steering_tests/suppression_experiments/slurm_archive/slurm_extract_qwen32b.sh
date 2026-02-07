#!/bin/bash
#SBATCH --job-name=suppress_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/suppress_qwen32b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/suppress_qwen32b_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "Extracting suppression vectors for Qwen 32B"
echo "============================================="

python steering_tests/suppression_experiments/extract_suppression_vectors.py \
    --model qwen32b \
    --n_samples 100 \
    --seed 42 \
    --output_dir steering_tests/suppression_experiments/vectors \
    --tensor_parallel 2

echo "Done!"
