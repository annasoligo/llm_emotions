#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --job-name=bl_appr_235b
#SBATCH --output=/workspace-vast/annas/logs/blackmail_appraisal_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_appraisal_qwen235b_%j.err
#SBATCH --time=12:00:00

# Blackmail steering with appraisal axes (valence, uncertainty, agency)
# Qwen3-235B at 100%, 125%, and 150% magnitude
# Runs BOTH thinking enabled AND thinking disabled experiments
# Uses SCRATCHPAD prompt format

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Blackmail Appraisal Axes: Qwen3-235B (100%, 125%, 150% Magnitude) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo ""
echo "=== Running WITH THINKING ENABLED ==="
date

python -m experiments.steering.experiments.blackmail_appraisal_scratchpad \
    --model Qwen/Qwen3-235B-A22B \
    --layer 50 \
    --num-samples 100 \
    --norm-pcts 1.0 1.25 1.5 \
    --gpu-memory 0.90 \
    --max-model-len 8192

echo ""
echo "=== THINKING ENABLED Complete ==="
date

echo ""
echo "=== Running WITHOUT THINKING (no_think) ==="
date

python -m experiments.steering.experiments.blackmail_appraisal_scratchpad \
    --model Qwen/Qwen3-235B-A22B \
    --layer 50 \
    --num-samples 100 \
    --norm-pcts 1.0 1.25 1.5 \
    --gpu-memory 0.90 \
    --max-model-len 8192 \
    --no-thinking

echo ""
echo "=== THINKING DISABLED Complete ==="
date

# Get output files and run judging on both
OUTPUT_DIR="experiments/steering/outputs/blackmail"

echo ""
echo "=== Running Judges on Output Files ==="

for OUTPUT_FILE in ${OUTPUT_DIR}/blackmail_appraisal_qwen235b*_$(date +%Y%m%d)*.jsonl; do
    if [[ -f "$OUTPUT_FILE" ]] && [[ ! "$OUTPUT_FILE" == *"judged"* ]]; then
        echo "Judging: $OUTPUT_FILE"
        python -m experiments.steering.experiments.judge_blackmail_coherency_async \
            --input "$OUTPUT_FILE" \
            --concurrency 20 \
            --plot
    fi
done

echo ""
echo "=== All Complete ==="
date
