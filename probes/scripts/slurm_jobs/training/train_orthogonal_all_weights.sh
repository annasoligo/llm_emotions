#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=ortho_all_%a
#SBATCH --output=/workspace-vast/annas/logs/%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/%A_%a.err
#SBATCH --array=0-4

# Train orthogonal probes with all tested orthogonality weights
# Tests ortho_weight = 1.0, 10.0, 100.0, 1000.0
# Array 0-4 = layers 10, 20, 30, 40, 50

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Configuration
LAYERS=(10 20 30 40 50)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Training orthogonal probes for layer $LAYER"
echo "All ortho weights: 1.0, 10.0, 100.0, 1000.0"
echo "=========================================="
echo ""

# Test all orthogonality weights
for ORTHO_WEIGHT in 1.0 10.0 100.0 1000.0; do
    echo "=== Orthogonality weight = $ORTHO_WEIGHT ==="

    # Raw
    echo "  [1/3] Raw (ortho=$ORTHO_WEIGHT)"
    python -m probes.scripts.training.train_orthogonal_conversation_probe \
        --layer $LAYER \
        --representation raw \
        --ortho-weight $ORTHO_WEIGHT \
        --device cuda

    # Global cPCA top 10
    echo "  [2/3] Global cPCA top 10 (ortho=$ORTHO_WEIGHT)"
    python -m probes.scripts.training.train_orthogonal_conversation_probe \
        --layer $LAYER \
        --representation global_cpca \
        --n-components 10 \
        --ortho-weight $ORTHO_WEIGHT \
        --device cuda

    # Regional cPCA top 10
    echo "  [3/3] Regional cPCA top 10 (ortho=$ORTHO_WEIGHT)"
    python -m probes.scripts.training.train_orthogonal_conversation_probe \
        --layer $LAYER \
        --representation regional_cpca \
        --n-components 10 \
        --ortho-weight $ORTHO_WEIGHT \
        --device cuda

    echo ""
done

echo "=========================================="
echo "Completed all orthogonal probes for layer $LAYER"
echo "=========================================="
