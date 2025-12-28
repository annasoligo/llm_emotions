#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=ortho1000_global
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err
#SBATCH --array=0-1

# Train orthogonal probes with ortho_weight=1000.0 for ALL layers (0-61)
# Global cPCA representations: top 10 and top 20

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Array index determines n_components
N_COMPONENTS=(10 20)
N_COMP=${N_COMPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Training orthogonal probes (ortho=1000.0)"
echo "All layers: 0-61"
echo "Representation: global_cpca top $N_COMP"
echo "=========================================="
echo ""

OUTPUT_DIR="outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_1000.0"
GLOBAL_CPCA_PATH="outputs/dimensionality_reduction/cpca/conversation_based/global/gemma-3-27b-it_cpca.npz"

mkdir -p "$OUTPUT_DIR"

# Train for all 62 layers (0-61)
for LAYER in {0..61}; do
    echo "=== Layer $LAYER (global cPCA top $N_COMP) ==="

    python -m probes.scripts.training.train_orthogonal_conversation_probe \
        --layer $LAYER \
        --representation global_cpca \
        --n-components $N_COMP \
        --global-cpca "$GLOBAL_CPCA_PATH" \
        --ortho-weight 1000.0 \
        --data outputs/data/activations/conversations2_combined.h5 \
        --output-dir "$OUTPUT_DIR" \
        --device cuda \
        --batch-size 64 \
        --max-epochs 200

    echo ""
done

echo "=========================================="
echo "Completed all layers for global cPCA top $N_COMP!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
