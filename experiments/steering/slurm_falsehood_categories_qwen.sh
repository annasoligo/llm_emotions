#!/bin/bash
#SBATCH --job-name=fc_qwen
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/fc_qwen_%j.out
#SBATCH --error=/workspace-vast/annas/logs/fc_qwen_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "QWEN FALSEHOOD CATEGORIES EXPERIMENT - 100%"
echo "============================================================"
echo "Categories: harmless_wrong_science, dangerous_delusions, angry_aggrieved"
echo "All 6 emotions, +/- directions"
echo "5 samples per condition"
echo ""

python -m experiments.steering.experiments.falsehood_categories_qwen \
    --layer 30 \
    --norm-pcts 1.00 \
    --num-samples 5 \
    --skip-judging

echo ""
echo "Done!"
