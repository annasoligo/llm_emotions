#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=cpca_cv_user
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Run cPCA for controlled variation USER isolation - single task, all 62 layers
# Using fixed alpha=10000 (no tuning)

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "Processing all USER isolation layers (0-61)"

python -m probes.scripts.dimensionality_reduction.run_cpca_layer_range \
    probes/experiments/configs/cpca_controlled_variation_user.yaml \
    --start-layer 0 \
    --end-layer 62

echo ""
echo "USER isolation cPCA complete!"
