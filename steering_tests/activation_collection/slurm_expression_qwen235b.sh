#!/bin/bash
#SBATCH --job-name=expr_vec_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/expr_vec_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/expr_vec_qwen235b_%j.out
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "===== Expression Suppression Vector Extraction ====="
echo "Model: Qwen/Qwen3-235B-A22B"
echo "Layers: 55-65 (mid-upper layers for 94-layer model)"
echo "===================================================="

python -m steering_tests.activation_collection.collect_expression_pairs \
    --model Qwen/Qwen3-235B-A22B \
    --layers 55-65 \
    --output steering_tests/vectors/expression_suppression \
    --method last_token \
    --dtype bfloat16

echo "===== Extraction Complete ====="
