#!/bin/bash
#SBATCH --job-name=expr_vec_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/expr_vec_qwen32b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/expr_vec_qwen32b_%j.out
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "===== Expression Suppression Vector Extraction ====="
echo "Model: Qwen/Qwen3-32B"
echo "Layers: 20-40 (mid layers for 64-layer model)"
echo "===================================================="

python -m steering_tests.activation_collection.collect_expression_pairs \
    --model Qwen/Qwen3-32B \
    --layers 20-40 \
    --output steering_tests/vectors/expression_suppression \
    --method last_token \
    --dtype bfloat16

echo "===== Extraction Complete ====="
