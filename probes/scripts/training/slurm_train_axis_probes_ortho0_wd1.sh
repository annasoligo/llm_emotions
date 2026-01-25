#!/bin/bash
#SBATCH --job-name=train_axis_ortho0_wd1
#SBATCH --output=logs/train_axis_ortho0_wd1_%j.log
#SBATCH --error=logs/train_axis_ortho0_wd1_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo

# Load secrets
echo "Loading secrets..."
source /workspace-vast/annas/.secrets/load_secrets.sh
echo "✓ Secrets loaded"
echo

# Activate virtual environment
echo "Activating virtual environment..."
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate
echo "✓ Virtual environment activated"
echo

echo "Training with ortho_weight=0.0 and weight_decay=1.0"
echo

# Train for each layer
for LAYER in 0 1 2 3 4 5 6 7 8 9 10; do
    echo "=========================================="
    echo "Training layer $LAYER with ortho_weight=0.0, weight_decay=1.0"
    echo "=========================================="
    echo

    python scripts/training/train_orthogonal_axis_probes.py \
        --data data/axis_paraphrases/activations.h5 \
        --output-dir results/axis_probes/ortho0.0_wd1.0 \
        --layer $LAYER \
        --ortho-weight 0.0 \
        --learning-rate 0.001 \
        --weight-decay 1.0 \
        --batch-size 64 \
        --n-epochs 100 \
        --patience 3 \
        --test-size 0.2 \
        --seed 42 \
        --device cuda

    echo
    echo "✓ Layer $LAYER complete"
    echo
done

echo
echo "=========================================="
echo "All layers complete for ortho_weight=0.0, weight_decay=1.0"
echo "Job finished at: $(date)"
echo "Exit code: $?"
echo "=========================================="
