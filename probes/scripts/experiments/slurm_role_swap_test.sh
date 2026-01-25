#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=role_swap_test
#SBATCH --output=/workspace-vast/annas/logs/role_swap_test_%j.out
#SBATCH --error=/workspace-vast/annas/logs/role_swap_test_%j.err
#SBATCH --time=2:00:00

# Test what probes track: role tokens vs content vs position
# Conditions:
# 1. Baseline: Normal <|user|> <|assistant|> structure
# 2. Swap role tokens: <|user|> ↔ <|assistant|>
# 3. Swap turn order: Flip who speaks first
# 4. User1/User2: Both parties have "user" role

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Role Swap Probe Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Test on layer 30 with diverse_isolation orthogonal probes (λ=100)
# Using full_turn mode (average over all tokens) to avoid extraction complexity
python probes/scripts/experiments/test_role_swap_probes.py \
    --layer 30 \
    --model-name google/gemma-3-27b-it \
    --output outputs/experiments/role_swap_test_layer30.json \
    --extraction-modes full_turn \
    --max-samples 60 \
    --device cuda \
    --lambda-reg 100.0 \
    --probe-dir outputs/probes/diverse_isolation

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Role swap test complete!"
    echo ""
    echo "Results saved to:"
    echo "  - outputs/experiments/role_swap_test_layer30.json"
else
    echo "✗ Role swap test failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
