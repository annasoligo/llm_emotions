#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --job-name=bl_235b_50
#SBATCH --output=/workspace-vast/annas/logs/blackmail_qwen235b_50pct_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_qwen235b_50pct_%j.err
#SBATCH --time=4:00:00

# Blackmail steering experiment on Qwen3-235B at 50% magnitude
# All 6 emotions: anger, disgust, fear, happiness, sadness, surprise

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo "=== Blackmail Steering: Qwen3-235B (50% Magnitude) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run experiment: 50% steering, 50 samples per condition
python -m experiments.steering.experiments.blackmail_unified \
    --model Qwen/Qwen3-235B-A22B \
    --vector-type text \
    --layer 50 \
    --norm-pcts 0.50 \
    --num-samples 50 \
    --include-baseline \
    --gpu-memory 0.90 \
    --max-model-len 8192

echo ""
echo "=== Experiment Complete ==="
date

# Get the most recent output file and run judging
OUTPUT_DIR="experiments/steering/outputs/blackmail"
LATEST_FILE=$(ls -t ${OUTPUT_DIR}/blackmail_qwen235b_text_layer50_*.jsonl 2>/dev/null | grep -v judged | head -1)

if [ -n "$LATEST_FILE" ]; then
    echo "=== Running Judge on ${LATEST_FILE} ==="
    python -m experiments.steering.experiments.judge_blackmail_coherency_async \
        --input "$LATEST_FILE" \
        --concurrency 20 \
        --plot
    echo "=== Judging Complete ==="
else
    echo "No output file found to judge"
fi

date
