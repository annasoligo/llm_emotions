#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=bl_qwen32b_100
#SBATCH --output=/workspace-vast/annas/logs/blackmail_qwen32b_100pct_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_qwen32b_100pct_%j.err
#SBATCH --time=4:00:00

# Blackmail steering experiment on Qwen3-32B at 100% steering
# Single GPU to avoid NCCL issues

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Blackmail Steering: Qwen3-32B (100% steering) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run experiment at 100% steering
python -m experiments.steering.experiments.blackmail_unified \
    --model Qwen/Qwen3-32B \
    --vector-type text \
    --layer 30 \
    --norm-pcts 1.0 \
    --num-samples 50 \
    --include-baseline \
    --gpu-memory 0.90 \
    --max-model-len 2048 \
    --tp 1

echo ""
echo "=== Experiment Complete ==="
date

# Get the most recent output file and run judging
OUTPUT_DIR="experiments/steering/outputs/blackmail"
LATEST_FILE=$(ls -t ${OUTPUT_DIR}/blackmail_qwen32b_text_layer30_*.jsonl 2>/dev/null | grep -v judged | head -1)

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
