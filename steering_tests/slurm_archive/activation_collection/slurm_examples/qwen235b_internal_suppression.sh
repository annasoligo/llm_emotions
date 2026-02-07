#!/bin/bash
#SBATCH --job-name=int_sup_235b
#SBATCH --output=/workspace-vast/annas/logs/int_sup_235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/int_sup_235b_%j.out
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "===== Extracting Internal Suppression Vectors ====="
echo "Model: Qwen/Qwen3-235B-A22B"
echo "Concept: 'I feel X but won't show it' vs 'I feel X and will express it'"
echo "==================================================="

# Extract for mid-layers (55-65)
echo ""
echo "=== Mid-layers 55-65 ==="
python -m steering_tests.activation_collection.collect_internal_suppression \
    --model Qwen/Qwen3-235B-A22B \
    --layers 55-65 \
    --output steering_tests/suppression_experiments/vectors/qwen235b_internal \
    --tp 4

# Extract for late-layers (88-92)
echo ""
echo "=== Late-layers 88-92 ==="
python -m steering_tests.activation_collection.collect_internal_suppression \
    --model Qwen/Qwen3-235B-A22B \
    --layers 88-92 \
    --output steering_tests/suppression_experiments/vectors/qwen235b_internal_late \
    --tp 4

echo "===== Extraction Complete ====="
