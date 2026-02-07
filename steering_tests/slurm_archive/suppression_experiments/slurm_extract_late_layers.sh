#!/bin/bash
#SBATCH --job-name=extract_late
#SBATCH --output=/workspace-vast/annas/logs/extract_late_%j.out
#SBATCH --error=/workspace-vast/annas/logs/extract_late_%j.out
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "===== Extracting Expression Vectors for Late Layers ====="
echo "Model: Qwen/Qwen3-235B-A22B"
echo "Layers: 88-92 (last 5 layers excluding final)"
echo "========================================================="

python -m steering_tests.suppression_experiments.extract_suppression_vectors \
    --model qwen235b \
    --output_dir steering_tests/suppression_experiments/vectors/qwen235b_expression_late \
    --tensor_parallel 4

echo "===== Extraction Complete ====="
