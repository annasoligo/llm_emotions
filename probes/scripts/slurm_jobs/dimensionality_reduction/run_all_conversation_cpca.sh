#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=cpca_conversations_all
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Run all conversation cPCA experiments (global + regional)
# This is a single SLURM job that runs everything sequentially

set -e

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "========================================"
echo "Conversation cPCA Pipeline"
echo "========================================"
echo

# Step 1: Global cPCA
echo "Step 1: Running global cPCA..."
if [ ! -f "data/activations/conversations2_combined.h5" ]; then
    echo "Error: conversations2_combined.h5 not found!"
    echo "Please run combine_activations.py first (with --type global)"
    exit 1
fi

python -m probes.scripts.run_cpca \
    probes/experiments/configs/cpca_conversations_global.yaml

echo "✓ Global cPCA complete"
echo

# Step 2: Create regional combined files
echo "Step 2: Creating regional combined files..."
python -m probes.scripts.combine_activations \
    --type regional \
    --emotional data/activations/conversations2.h5 \
    --neutral data/activations/conversations2_neutral.h5 \
    --output-dir data/activations/regional

echo "✓ Regional combined files created"
echo

# Step 3: Run regional cPCA for each region
echo "Step 3: Running regional cPCA..."

REGIONS=("user" "asst" "special1" "special2")

for region in "${REGIONS[@]}"; do
    echo "  Processing region: ${region}"

    # Create config for this region
    CONFIG_FILE="probes/experiments/configs/cpca_conversations_regional_${region}.yaml"
    cat > "$CONFIG_FILE" << EOF
# cPCA configuration for conversation data (regional: ${region})
name: cpca_conversations_regional_${region}
output_dir: /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_${region}
seed: 42
device: cuda
data_path: /workspace-vast/annas/git/research-tools/data/activations/regional/conversations2_combined_${region}.h5
model_name: google/gemma-3-27b-it
alpha: null
alpha_range: [100, 10000]
n_alphas: 10
n_components: 50
use_diffs: true
EOF

    # Run cPCA for this region
    python -m probes.scripts.run_cpca "$CONFIG_FILE"

    echo "  ✓ Region ${region} complete"
done

echo
echo "========================================"
echo "All cPCA experiments complete!"
echo "========================================"
echo "Results:"
echo "  Global: probes/results/cpca_conversations_global/"
for region in "${REGIONS[@]}"; do
    echo "  Regional (${region}): probes/results/cpca_conversations_regional_${region}/"
done
