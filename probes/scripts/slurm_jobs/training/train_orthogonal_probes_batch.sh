#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=ortho_probe_%a
#SBATCH --output=/workspace-vast/annas/logs/%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/%A_%a.err
#SBATCH --array=0-4

# Train orthogonal conversation emotion probes
# Each array task trains all representations for one layer
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
ORTHO_WEIGHT=1.0

echo "=========================================="
echo "Training orthogonal probes for layer $LAYER"
echo "Orthogonality weight: $ORTHO_WEIGHT"
echo "=========================================="
echo ""

# 1. Raw
echo "=== [1/7] Raw mean activations (ortho) ==="
python -m probes.scripts.training.train_orthogonal_conversation_probe \
    --layer $LAYER \
    --representation raw \
    --ortho-weight $ORTHO_WEIGHT \
    --device cuda
echo ""

# 2-4. Global cPCA (top 3, 5, 10)
for N_COMP in 3 5 10; do
    echo "=== Global cPCA top $N_COMP (ortho) ==="
    python -m probes.scripts.training.train_orthogonal_conversation_probe \
        --layer $LAYER \
        --representation global_cpca \
        --n-components $N_COMP \
        --ortho-weight $ORTHO_WEIGHT \
        --device cuda
    echo ""
done

# 5-7. Regional cPCA (top 3, 5, 10 per region)
for N_COMP in 3 5 10; do
    echo "=== Regional cPCA top $N_COMP per region (= $((N_COMP * 4)) total, ortho) ==="
    python -m probes.scripts.training.train_orthogonal_conversation_probe \
        --layer $LAYER \
        --representation regional_cpca \
        --n-components $N_COMP \
        --ortho-weight $ORTHO_WEIGHT \
        --device cuda
    echo ""
done

echo "=========================================="
echo "Completed all 7 orthogonal probe representations for layer $LAYER"
echo "=========================================="
