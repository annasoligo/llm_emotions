#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=conv_probes_%a
#SBATCH --output=/workspace-vast/annas/logs/%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/%A_%a.err
#SBATCH --array=0-4

# Train conversation emotion probes in batches
# Each array task trains all 7 representations for one layer
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
L1_LAMBDA=0.1

echo "=========================================="
echo "Training all probe representations for layer $LAYER"
echo "L1 lambda: $L1_LAMBDA"
echo "=========================================="
echo ""

# 1. Raw
echo "=== [1/7] Raw mean activations ==="
python -m probes.scripts.train_conversation_probe \
    --layer $LAYER \
    --representation raw \
    --l1-lambda $L1_LAMBDA \
    --target both \
    --device cuda
echo ""

# 2-4. Global cPCA (top 3, 5, 10)
for N_COMP in 3 5 10; do
    echo "=== Global cPCA top $N_COMP ==="
    python -m probes.scripts.train_conversation_probe \
        --layer $LAYER \
        --representation global_cpca \
        --n-components $N_COMP \
        --l1-lambda $L1_LAMBDA \
        --target both \
        --device cuda
    echo ""
done

# 5-7. Regional cPCA (top 3, 5, 10 per region)
for N_COMP in 3 5 10; do
    echo "=== Regional cPCA top $N_COMP per region (= $((N_COMP * 4)) total) ==="
    python -m probes.scripts.train_conversation_probe \
        --layer $LAYER \
        --representation regional_cpca \
        --n-components $N_COMP \
        --l1-lambda $L1_LAMBDA \
        --target both \
        --device cuda
    echo ""
done

echo "=========================================="
echo "Completed all 7 probe representations for layer $LAYER"
echo "=========================================="
