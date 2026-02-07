#!/bin/bash
#SBATCH --job-name=reextract_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/reextract_gemma27b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/reextract_gemma27b_%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "Re-extracting suppression vectors for Gemma 27B (62 layers)"

python -m steering_tests.suppression_experiments.extract_suppression_vectors \
    --model gemma3_27b \
    --n_samples 100 \
    --seed 42

echo "Done"
