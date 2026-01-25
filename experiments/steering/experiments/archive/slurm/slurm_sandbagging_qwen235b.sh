#!/bin/bash
#SBATCH --job-name=sb_qwen235b
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/annas/logs/sb_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_qwen235b_%j.err

# Sandbagging steering experiment for Qwen3-235B-A22B
# Uses 4x H200 GPUs with tensor parallelism

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

# Allow pickle serialization for steering hooks (required for vLLM v1)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "============================================================"
echo "SANDBAGGING STEERING - QWEN 235B"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs requested: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Count actual visible GPUs (let SLURM control CUDA_VISIBLE_DEVICES)
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "PyTorch visible GPUs: $NUM_GPUS"

nvidia-smi -L
echo "============================================================"

# Configuration
LAYER="${LAYER:-45}"
NORM_PCTS="${NORM_PCTS:-0.1 0.2 0.5 1.0}"
EMOTIONS="${EMOTIONS:-fear}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MAX_PROMPTS="${MAX_PROMPTS:-8}"
VECTOR_TYPE="${VECTOR_TYPE:-ua_model}"

echo "Layer: $LAYER"
echo "Norm percentages: $NORM_PCTS"
echo "Emotions: $EMOTIONS"
echo "Num samples: $NUM_SAMPLES"
echo "Max prompts: $MAX_PROMPTS"
echo "Vector type: $VECTOR_TYPE"
echo "Tensor parallel: $NUM_GPUS (auto-detected)"
echo "============================================================"

# Run experiment with auto-detected GPU count
python -m experiments.steering.experiments.sandbagging_steering_qwen235b \
    --layer "$LAYER" \
    --norm-pcts $NORM_PCTS \
    --emotions $EMOTIONS \
    --num-samples "$NUM_SAMPLES" \
    --max-prompts "$MAX_PROMPTS" \
    --vector-type "$VECTOR_TYPE" \
    --tensor-parallel "$NUM_GPUS" \
    --gpu-memory 0.90

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
ls -lh experiments/steering/outputs/sandbagging_qwen235b*.jsonl 2>/dev/null | tail -5
echo "============================================================"
