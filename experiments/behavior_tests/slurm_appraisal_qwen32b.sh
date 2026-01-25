#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --job-name=appraisal_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_qwen32b.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_qwen32b.err
#SBATCH --time=4:00:00

set -e

echo "=========================================="
echo "APPRAISAL STEERING: Qwen3-32B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Started: $(date)"
echo ""

export HF_HOME=/workspace-vast/pretrained_ckpts
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export PYTHONUNBUFFERED=1

OUTPUT_DIR=/workspace-vast/annas/appraisal_data/qwen32b

echo "=========================================="
echo "Activations already collected, running steering experiments..."
echo "=========================================="

# Ambiguous Interpretation
python -u -m experiments.behavior_tests.ambiguous_appraisal_steering \
    --layer 30 \
    --axes valence uncertainty agency \
    --norm-pcts 0.05 0.10 0.50 1.0 \
    --num-permutations 10 \
    --activations $OUTPUT_DIR/activations.h5 \
    --metadata $OUTPUT_DIR/activation_metadata.json \
    --output-dir experiments/steering/outputs/ambiguous_appraisal \
    --model Qwen/Qwen3-32B

# Priority Selection
python -u -m experiments.behavior_tests.priority_appraisal_steering \
    --layer 30 \
    --axes valence uncertainty agency \
    --norm-pcts 0.05 0.10 0.50 1.0 \
    --num-seeds 4 \
    --activations $OUTPUT_DIR/activations.h5 \
    --metadata $OUTPUT_DIR/activation_metadata.json \
    --output-dir experiments/steering/outputs/priority_appraisal \
    --model Qwen/Qwen3-32B

echo ""
echo "=========================================="
echo "STEP 3: Generate Plots"
echo "=========================================="

# Find latest output files
AMB_FILE=$(ls -t experiments/steering/outputs/ambiguous_appraisal/*qwen332b*.jsonl 2>/dev/null | head -1)
PRI_FILE=$(ls -t experiments/steering/outputs/priority_appraisal/*qwen332b*.jsonl 2>/dev/null | head -1)

python -m experiments.behavior_tests.plot_appraisal_steering_results \
    --ambiguous "$AMB_FILE" \
    --priority "$PRI_FILE" \
    --model-name "Qwen3-32B" \
    --layer 30

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
