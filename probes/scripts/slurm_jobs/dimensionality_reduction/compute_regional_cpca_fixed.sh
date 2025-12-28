#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=regional_cpca
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Compute regional cPCA for conversation activations using existing config files
# Config files already exist at probes/experiments/configs/cpca_conversations_regional_*.yaml

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Computing Regional cPCA"
echo "Using existing config files with alpha=5.0"
echo "N components: 64"
echo "Regions: user, asst, special1, special2"
echo "=========================================="
echo ""

# Run cPCA for each region using existing config files
for REGION in user asst special1 special2; do
    echo "=== Computing cPCA for region: $REGION ==="

    CONFIG_FILE="probes/experiments/configs/cpca_conversations_regional_${REGION}.yaml"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi

    python -m probes.scripts.dimensionality_reduction.run_cpca "$CONFIG_FILE"

    echo ""
done

echo "=========================================="
echo "Regional cPCA computation complete!"
echo "Results saved to: probes/results/cpca_conversations_regional_*/"
echo "=========================================="
