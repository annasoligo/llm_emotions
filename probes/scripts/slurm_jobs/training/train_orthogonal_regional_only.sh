#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=ortho_reg_%a
#SBATCH --output=/workspace-vast/annas/logs/%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/%A_%a.err
#SBATCH --array=0-4

# Train orthogonal regional cPCA probes only
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
echo "Training orthogonal REGIONAL probes for layer $LAYER"
echo "Orthogonality weight: $ORTHO_WEIGHT"
echo "=========================================="
echo ""

# Regional cPCA (top 3, 5, 10 per region)
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
echo "Completed all 3 orthogonal regional probe representations for layer $LAYER"
echo "=========================================="
