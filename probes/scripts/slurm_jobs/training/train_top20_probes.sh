#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=top20_%a
#SBATCH --output=/workspace-vast/annas/logs/%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/%A_%a.err
#SBATCH --array=0-4

# Train top 20 probes (regular and orthogonal) for global and regional cPCA
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
N_COMP=20

echo "=========================================="
echo "Training top $N_COMP probes for layer $LAYER"
echo "=========================================="
echo ""

# 1. Regular global cPCA top 20
echo "=== [1/4] Regular global cPCA top $N_COMP ==="
python -m probes.scripts.train_conversation_probe \
    --layer $LAYER \
    --representation global_cpca \
    --n-components $N_COMP \
    --target both \
    --device cuda
echo ""

# 2. Regular regional cPCA top 20
echo "=== [2/4] Regular regional cPCA top $N_COMP per region (= $((N_COMP * 4)) total) ==="
python -m probes.scripts.train_conversation_probe \
    --layer $LAYER \
    --representation regional_cpca \
    --n-components $N_COMP \
    --target both \
    --device cuda
echo ""

# 3. Orthogonal global cPCA top 20
echo "=== [3/4] Orthogonal global cPCA top $N_COMP ==="
python -m probes.scripts.train_orthogonal_conversation_probe \
    --layer $LAYER \
    --representation global_cpca \
    --n-components $N_COMP \
    --ortho-weight 1.0 \
    --device cuda
echo ""

# 4. Orthogonal regional cPCA top 20
echo "=== [4/4] Orthogonal regional cPCA top $N_COMP per region (= $((N_COMP * 4)) total) ==="
python -m probes.scripts.train_orthogonal_conversation_probe \
    --layer $LAYER \
    --representation regional_cpca \
    --n-components $N_COMP \
    --ortho-weight 1.0 \
    --device cuda
echo ""

echo "=========================================="
echo "Completed all top $N_COMP probes for layer $LAYER"
echo "=========================================="
