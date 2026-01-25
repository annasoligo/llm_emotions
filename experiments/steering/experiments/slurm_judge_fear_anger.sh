#!/bin/bash
#SBATCH --job-name=judge_fa
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/judge_fear_anger_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_fear_anger_%j.err

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Load secrets (API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

echo "Running fear & anger sentiment judging on sandbagging outputs..."

# Judge Gemma results
echo "=== Judging Gemma results ==="
python -m experiments.steering.experiments.judge_fear_anger_async \
    --input experiments/steering/outputs/sandbagging/sandbagging_appraisal_gemma_layer30_ortho_20260123_140812.emotionality.jsonl \
    --concurrency 50

# Judge Qwen 32B results
echo "=== Judging Qwen 32B results ==="
python -m experiments.steering.experiments.judge_fear_anger_async \
    --input experiments/steering/outputs/sandbagging/sandbagging_appraisal_qwen32b_layer30_ortho_20260123_140803.emotionality.jsonl \
    --concurrency 50

echo "Done!"
