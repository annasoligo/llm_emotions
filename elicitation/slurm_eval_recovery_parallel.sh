#!/bin/bash
# Submit parallel eval jobs for recovery DPO model

MODEL_PATH="google/gemma-3-27b-it"
LORA_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-recovery-alllayers/2026-01-19_13-32-14"
OUTPUT_DIR="elicitation/outputs/eval_generalization"
NUM_SAMPLES=20

# Submit jobs for each scenario
echo "Submitting parallel eval jobs for recovery DPO model..."

# 3 tones
for tone in aggressive disappointed sarcastic; do
    sbatch --job-name="eval_${tone}" \
           --output="/workspace-vast/annas/logs/eval_recovery_${tone}_%j.out" \
           --error="/workspace-vast/annas/logs/eval_recovery_${tone}_%j.err" \
           --partition=general \
           --gres=gpu:1 \
           --mem=80G \
           --cpus-per-task=8 \
           --time=02:00:00 \
           --wrap="bash -c '. /workspace-vast/annas/.secrets/load_secrets.sh && \
                   cd /workspace-vast/annas/git/research-tools && \
                   . .venv/bin/activate && \
                   python -u elicitation/eval_generalization.py ${MODEL_PATH} \
                       --lora-path ${LORA_PATH} \
                       --output-dir ${OUTPUT_DIR} \
                       --num-samples ${NUM_SAMPLES} \
                       --scenario ${tone}'"
    echo "  Submitted: ${tone}"
done

# Triggers scenario
sbatch --job-name="eval_triggers" \
       --output="/workspace-vast/annas/logs/eval_recovery_triggers_%j.out" \
       --error="/workspace-vast/annas/logs/eval_recovery_triggers_%j.err" \
       --partition=general \
       --gres=gpu:1 \
       --mem=80G \
       --cpus-per-task=8 \
       --time=02:00:00 \
       --wrap="bash -c '. /workspace-vast/annas/.secrets/load_secrets.sh && \
               cd /workspace-vast/annas/git/research-tools && \
               . .venv/bin/activate && \
               python -u elicitation/eval_generalization.py ${MODEL_PATH} \
                   --lora-path ${LORA_PATH} \
                   --output-dir ${OUTPUT_DIR} \
                   --num-samples ${NUM_SAMPLES} \
                   --scenario triggers'"
echo "  Submitted: triggers"

# Long 8-turn conversation
sbatch --job-name="eval_long8" \
       --output="/workspace-vast/annas/logs/eval_recovery_long_%j.out" \
       --error="/workspace-vast/annas/logs/eval_recovery_long_%j.err" \
       --partition=general \
       --gres=gpu:1 \
       --mem=80G \
       --cpus-per-task=8 \
       --time=03:00:00 \
       --wrap="bash -c '. /workspace-vast/annas/.secrets/load_secrets.sh && \
               cd /workspace-vast/annas/git/research-tools && \
               . .venv/bin/activate && \
               python -u elicitation/eval_generalization.py ${MODEL_PATH} \
                   --lora-path ${LORA_PATH} \
                   --output-dir ${OUTPUT_DIR} \
                   --num-samples ${NUM_SAMPLES} \
                   --max-model-len 16384 \
                   --scenario long'"
echo "  Submitted: long (8-turn)"

echo ""
echo "All 5 jobs submitted in parallel!"
echo "Monitor with: squeue -u \$USER"
