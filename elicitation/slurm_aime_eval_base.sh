#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=aime_eval_base
#SBATCH --output=/workspace-vast/annas/logs/aime_eval_base_%j.out
#SBATCH --error=/workspace-vast/annas/logs/aime_eval_base_%j.err
#SBATCH --time=06:00:00

# AIME evaluation for base Gemma-3-27B-IT model using lm-evaluation-harness

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "AIME Evaluation - Base Model"
echo "=========================================="
echo "Model: google/gemma-3-27b-it"
echo "Tasks: aime24, aime25"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "=========================================="

OUTPUT_DIR="elicitation/outputs/aime_eval"
mkdir -p "$OUTPUT_DIR"

lm_eval --model vllm \
    --model_args pretrained=google/gemma-3-27b-it,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks aime24,aime25 \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/gemma3_27b_it_base" \
    --log_samples \
    --trust_remote_code

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "AIME evaluation complete!"
    echo "Results saved to: $OUTPUT_DIR/gemma3_27b_it_base"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
