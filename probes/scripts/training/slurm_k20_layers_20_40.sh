#!/bin/bash
#SBATCH --job-name=k20_l20-40
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/train_k20_layers_20_40_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/train_k20_layers_20_40_%A.err
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

echo "========================================"
echo "Training K=20 Layers 20-40 (21 layers)"
echo "========================================"

for LAYER in {20..40}; do
    echo ""
    echo "========================================"
    echo "Training Layer $LAYER (K=20)"
    echo "========================================"
    
    python probes/scripts/training/train_multi_orthogonal_conversation_probes.py \
        --layer $LAYER \
        --n-sets 20 \
        --ortho-weight 100000.0 \
        --max-epochs 300 \
        --patience 3 \
        --learning-rate 0.001 \
        --batch-size 32 \
        --seed 42 \
        --device cuda
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for layer $LAYER"
        exit 1
    fi
    
    echo "✓ Layer $LAYER completed!"
done

echo ""
echo "========================================"
echo "✓ All K=20 training completed!"
echo "========================================"
