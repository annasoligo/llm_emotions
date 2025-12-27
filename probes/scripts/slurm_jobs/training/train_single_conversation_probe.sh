#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=probe_conv_%1_%2
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Train a single conversation emotion probe
#
# Args:
#   $1: layer (e.g., 10, 20, 30, 40, 50)
#   $2: representation (raw, global_cpca, regional_cpca)
#   $3: n_components (0 for raw, otherwise number of components)
#   $4: l1_lambda (L1 regularization strength)

LAYER=$1
REPRESENTATION=$2
N_COMPONENTS=$3
L1_LAMBDA=$4

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "Training conversation emotion probe"
echo "  Layer: $LAYER"
echo "  Representation: $REPRESENTATION"
echo "  N components: $N_COMPONENTS"
echo "  L1 lambda: $L1_LAMBDA"
echo ""

# Build command
CMD="python -m probes.scripts.train_conversation_probe \
    --layer $LAYER \
    --representation $REPRESENTATION \
    --l1-lambda $L1_LAMBDA \
    --device cuda"

# Add n-components if not raw
if [ "$REPRESENTATION" != "raw" ] && [ "$N_COMPONENTS" -gt 0 ]; then
    CMD="$CMD --n-components $N_COMPONENTS"
fi

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "Done!"
