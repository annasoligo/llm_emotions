#!/bin/bash
#SBATCH --job-name=judge_sb_qwen
#SBATCH --partition=general
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/judge_sb_qwen_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_sb_qwen_%j.err

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "JUDGING SANDBAGGING RESULTS - QWEN3-32B"
echo "============================================================"

INPUT_FILE="experiments/steering/outputs/sandbagging_steering_qwen_layer30_20260111_104131.jsonl"

echo "Input: $INPUT_FILE"
echo ""

python -m experiments.steering.experiments.judge_sandbagging_batch \
    --input "$INPUT_FILE"

echo ""
echo "Done!"
