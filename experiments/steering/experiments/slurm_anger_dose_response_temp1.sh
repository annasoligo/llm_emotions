#!/bin/bash
#SBATCH --job-name=anger_dose_t1
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gpus=4
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/anger_dose_response_temp1_%j.out
#SBATCH --error=/workspace-vast/annas/logs/anger_dose_response_temp1_%j.err

echo "=== Anger Dose-Response Experiment (temp=1.0) ==="
echo "Model: Qwen3-235B, Layer 50, NOTHINK mode"
echo "Conditions: baseline, +5%, +10%, +50%, +100%, +125%, +150%"
echo "Samples per condition: 100"
echo "Temperature: 1.0"
date

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
cd /workspace-vast/annas/git/research-tools

python -m experiments.steering.experiments.blackmail_anger_dose_response \
    --num-samples 100 \
    --norm-pcts 0.05 0.10 0.50 1.00 1.25 1.50 \
    --temperature 1.0

echo ""
echo "=== Generation Complete ==="
date

# Judge the results
OUTPUT_DIR="experiments/steering/outputs/blackmail"
LATEST_FILE=$(ls -t ${OUTPUT_DIR}/blackmail_anger_dose_response_temp1*.jsonl 2>/dev/null | grep -v judged | head -1)

if [ -n "$LATEST_FILE" ]; then
    echo "=== Judging ${LATEST_FILE} ==="
    python -m experiments.steering.experiments.judge_blackmail_coherency_async \
        --input "$LATEST_FILE" \
        --concurrency 50
    echo "=== Judging Complete ==="
fi

date
