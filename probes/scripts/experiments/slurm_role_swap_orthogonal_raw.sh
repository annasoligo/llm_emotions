#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=role_swap_ortho_raw
#SBATCH --output=/workspace-vast/annas/logs/role_swap_ortho_raw_%j.out
#SBATCH --error=/workspace-vast/annas/logs/role_swap_ortho_raw_%j.err
#SBATCH --time=2:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Role Swap Test - Orthogonal Raw Probes"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Test on layer 30 with orthogonal raw probes (ortho_weight=1000.0)
python probes/scripts/experiments/test_role_swap_orthogonal_raw.py \
    --layer 30 \
    --model-name unsloth/gemma-3-27b-it \
    --output outputs/experiments/role_swap_orthogonal_raw_layer30.json \
    --max-samples 60 \
    --device cuda \
    --ortho-weight 1000.0

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Role swap test complete!"
    echo ""
    echo "Results saved to:"
    echo "  - outputs/experiments/role_swap_orthogonal_raw_layer30.json"
else
    echo "✗ Role swap test failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
