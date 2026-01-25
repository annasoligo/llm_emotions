#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=math_base
#SBATCH --output=/workspace-vast/annas/logs/math_base_%j.out
#SBATCH --error=/workspace-vast/annas/logs/math_base_%j.err
#SBATCH --time=06:00:00

# MATH evaluation for base model

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "MATH - Base Model"
echo "=========================================="
echo "Model: google/gemma-3-27b-it"
echo "Tasks: leaderboard_math_hard"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

OUTPUT_DIR="elicitation/outputs/math_eval"
mkdir -p "$OUTPUT_DIR"

lm_eval --model vllm \
    --model_args pretrained=google/gemma-3-27b-it,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True \
    --tasks leaderboard_math_hard \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/gemma3_27b_it_base" \
    --log_samples \
    --trust_remote_code

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation complete!"
    echo "Results saved to: $OUTPUT_DIR/gemma3_27b_it_base"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
