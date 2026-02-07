#!/bin/bash
# Submit all wildchat early generation jobs

DATA_FILE="experiments/prefill_scaled/prepared_wildchat_early_20260127_202312.json"

for MODEL_FAMILY in gemma27b gemma12b qwen32b olmo32b; do
    for MODEL_TYPE in instruct base; do
        echo "Submitting: $MODEL_FAMILY $MODEL_TYPE"
        sbatch --job-name="wc_early_${MODEL_FAMILY}_${MODEL_TYPE}" \
               --output="experiments/prefill_scaled/logs/wc_early_%x_%j.log" \
               --error="experiments/prefill_scaled/logs/wc_early_%x_%j.err" \
               --partition=general \
               --gres=gpu:1 \
               --mem=80G \
               --cpus-per-task=8 \
               --time=4:00:00 \
               --wrap="cd /workspace-vast/annas/git/research-tools && source /workspace-vast/annas/.secrets/load_secrets.sh && source .venv/bin/activate && python experiments/prefill_scaled/run_wildchat_early.py --model-family $MODEL_FAMILY --model-type $MODEL_TYPE --data $DATA_FILE"
    done
done

echo ""
echo "Submitted 8 jobs. Monitor with: squeue -u annas"
