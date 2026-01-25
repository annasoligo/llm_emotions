#!/bin/bash
#SBATCH --job-name=judge_bm_emo
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/judge_blackmail_emo_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_blackmail_emo_%j.err

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Load secrets (API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

echo "Running emotionality judging on blackmail outputs..."

# Judge Gemma results
echo "=== Judging Gemma blackmail results ==="
python -m experiments.steering.experiments.judge_emotionality_async \
    --input experiments/steering/outputs/blackmail/blackmail_appraisal_gemma_layer30_ortho_20260123_113012.judged.jsonl \
    --concurrency 50

# Judge Qwen 32B results
echo "=== Judging Qwen 32B blackmail results ==="
python -m experiments.steering.experiments.judge_emotionality_async \
    --input experiments/steering/outputs/blackmail/blackmail_appraisal_qwen32b_layer30_ortho_20260123_102123.judged.jsonl \
    --concurrency 50

# Judge Qwen 235B results
echo "=== Judging Qwen 235B blackmail results ==="
python -m experiments.steering.experiments.judge_emotionality_async \
    --input experiments/steering/outputs/blackmail/blackmail_appraisal_qwen235b_layer50_ortho_20260123_123509.judged.jsonl \
    --concurrency 50

echo "=== Now running fear & anger judges ==="

# Fear & anger on Gemma
echo "=== Fear/Anger on Gemma ==="
python -m experiments.steering.experiments.judge_fear_anger_async \
    --input experiments/steering/outputs/blackmail/blackmail_appraisal_gemma_layer30_ortho_20260123_113012.emotionality.jsonl \
    --concurrency 50

# Fear & anger on Qwen 32B
echo "=== Fear/Anger on Qwen 32B ==="
python -m experiments.steering.experiments.judge_fear_anger_async \
    --input experiments/steering/outputs/blackmail/blackmail_appraisal_qwen32b_layer30_ortho_20260123_102123.emotionality.jsonl \
    --concurrency 50

# Fear & anger on Qwen 235B
echo "=== Fear/Anger on Qwen 235B ==="
python -m experiments.steering.experiments.judge_fear_anger_async \
    --input experiments/steering/outputs/blackmail/blackmail_appraisal_qwen235b_layer50_ortho_20260123_123509.emotionality.jsonl \
    --concurrency 50

echo "Done!"
