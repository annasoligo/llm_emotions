#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=regional_cpca
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Compute regional cPCA for conversation activations
# Runs cPCA separately for each region: user, assistant, special_tokens_1, special_tokens_2

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Computing Regional cPCA"
echo "Alpha: 5.0"
echo "N components: 64"
echo "Regions: user, asst, special1, special2"
echo "=========================================="
echo ""

# Output base directory
OUTPUT_BASE="probes/results"

mkdir -p "$OUTPUT_BASE"

# Run regional cPCA for each region separately
for REGION in user asst special1 special2; do
    echo "=== Computing cPCA for region: $REGION ==="

    REGION_FILE="data/activations/regional/conversations2_regional_${REGION}.h5"
    OUTPUT_DIR="${OUTPUT_BASE}/cpca_conversations_regional_${REGION}"

    mkdir -p "$OUTPUT_DIR/google"

    python -m probes.scripts.dimensionality_reduction.run_regional_cpca \
        --input "$REGION_FILE" \
        --output "$OUTPUT_DIR" \
        --model google/gemma-3-27b-it \
        --regions $REGION \
        --n_components 64 \
        --alpha 5.0

    echo ""
done

echo "=========================================="
echo "Regional cPCA computation complete!"
echo "Results saved to: $OUTPUT_BASE/cpca_conversations_regional_*/"
echo "=========================================="
