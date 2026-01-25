#!/bin/bash
#SBATCH --job-name=train_axis_probes
#SBATCH --output=logs/train_axis_ortho%a_%j.log
#SBATCH --error=logs/train_axis_ortho%a_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-1  # 0=ortho_weight 0, 1=ortho_weight 100

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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

# Set ortho_weight based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    ORTHO_WEIGHT=0.0
    echo "Training with ortho_weight=0.0 (independent baseline)"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    ORTHO_WEIGHT=100.0
    echo "Training with ortho_weight=100.0 (strong orthogonality)"
fi
echo

# Train for each layer
for LAYER in 0 1 2 3 4 5 6 7 8 9 10; do
    echo "=========================================="
    echo "Training layer $LAYER with ortho_weight=$ORTHO_WEIGHT"
    echo "=========================================="
    echo

    python scripts/training/train_orthogonal_axis_probes.py \
        --data data/axis_paraphrases/activations.h5 \
        --output-dir results/axis_probes/ortho${ORTHO_WEIGHT} \
        --layer $LAYER \
        --ortho-weight $ORTHO_WEIGHT \
        --learning-rate 0.001 \
        --weight-decay 0.0 \
        --batch-size 64 \
        --n-epochs 100 \
        --patience 10 \
        --test-size 0.2 \
        --seed 42 \
        --device cuda

    echo
    echo "✓ Layer $LAYER complete"
    echo
done

echo
echo "=========================================="
echo "All layers complete for ortho_weight=$ORTHO_WEIGHT"
echo "Job finished at: $(date)"
echo "Exit code: $?"
echo "=========================================="
