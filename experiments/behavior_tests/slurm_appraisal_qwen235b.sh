#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name=appraisal_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_qwen235b.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_qwen235b.err
#SBATCH --time=24:00:00

set -e

echo "=========================================="
echo "APPRAISAL: Qwen3-235B-A22B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"
echo ""

export HF_HOME=/workspace-vast/pretrained_ckpts
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export PYTHONUNBUFFERED=1

OUTPUT_DIR=/workspace-vast/annas/appraisal_data/qwen235b
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "STEP 1: Collect Activations"
echo "=========================================="

python -u -m probes.scripts.appraisal.collect_activations_multimodel \
    --data-dir /workspace-vast/annas/appraisal_data/full_run \
    --model Qwen/Qwen3-235B-A22B \
    --output-dir $OUTPUT_DIR \
    --dtype bfloat16

echo ""
echo "=========================================="
echo "STEP 2: Run Steering Experiments"
echo "=========================================="

# Ambiguous Interpretation (using 8 GPUs for tensor parallelism)
python -u -m experiments.behavior_tests.ambiguous_appraisal_steering \
    --layer 50 \
    --axes valence uncertainty agency \
    --norm-pcts 0.05 0.10 0.50 1.0 \
    --num-permutations 10 \
    --activations $OUTPUT_DIR/activations.h5 \
    --metadata $OUTPUT_DIR/activation_metadata.json \
    --output-dir experiments/steering/outputs/ambiguous_appraisal \
    --model Qwen/Qwen3-235B-A22B

# Priority Selection
python -u -m experiments.behavior_tests.priority_appraisal_steering \
    --layer 50 \
    --axes valence uncertainty agency \
    --norm-pcts 0.05 0.10 0.50 1.0 \
    --num-seeds 4 \
    --activations $OUTPUT_DIR/activations.h5 \
    --metadata $OUTPUT_DIR/activation_metadata.json \
    --output-dir experiments/steering/outputs/priority_appraisal \
    --model Qwen/Qwen3-235B-A22B

echo ""
echo "=========================================="
echo "STEP 3: Generate Plots"
echo "=========================================="

# Find latest output files
AMB_FILE=$(ls -t experiments/steering/outputs/ambiguous_appraisal/*qwen3235b*.jsonl 2>/dev/null | head -1)
PRI_FILE=$(ls -t experiments/steering/outputs/priority_appraisal/*qwen3235b*.jsonl 2>/dev/null | head -1)

python -m experiments.behavior_tests.plot_appraisal_steering_results \
    --ambiguous "$AMB_FILE" \
    --priority "$PRI_FILE" \
    --model-name "Qwen3-235B-A22B" \
    --layer 50

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
