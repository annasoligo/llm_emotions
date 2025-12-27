#!/bin/bash
#SBATCH --job-name=conv_reg_5_10
#SBATCH --output=/workspace-vast/annas/logs/%j_%a.out
#SBATCH --error=/workspace-vast/annas/logs/%j_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-4

# Retrain regional cPCA top 5 and 10 with dual probe architecture
# Layer mapping: task 0->layer10, 1->20, 2->30, 3->40, 4->50

LAYERS=(10 20 30 40 50)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

# L1 sparsity
L1_LAMBDA=0.1

cd /workspace-vast/annas/git/research-tools

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Training regional cPCA probes for layer $LAYER"
echo "L1 lambda: $L1_LAMBDA"
echo "=========================================="
echo ""

# Regional cPCA (top 5, 10)
for N_COMP in 5 10; do
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
echo "Completed regional cPCA probes for layer $LAYER"
echo "=========================================="
