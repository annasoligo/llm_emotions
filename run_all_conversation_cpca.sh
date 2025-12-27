#!/bin/bash
# Complete workflow for conversation cPCA (global + all regional)

set -e  # Exit on error

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "========================================"
echo "Conversation cPCA Pipeline"
echo "========================================"
echo

# Step 1: Run global cPCA
echo "Step 1: Running global cPCA..."
if [ ! -f "data/activations/conversations2_combined.h5" ]; then
    echo "Error: conversations2_combined.h5 not found!"
    exit 1
fi

echo "Submitting global cPCA job..."
GLOBAL_JOB=$(sbatch --parsable probes/scripts/slurm_jobs/run_conversation_cpca_global.sh)
echo "  Global cPCA: Job $GLOBAL_JOB"

# Step 2: Create regional combined files
echo
echo "Step 2: Creating regional combined files..."
python -m probes.scripts.combine_activations \
    --type regional \
    --emotional data/activations/conversations2.h5 \
    --neutral data/activations/conversations2_neutral.h5 \
    --output-dir data/activations/regional

# Step 3: Submit regional cPCA jobs for each region
echo
echo "Step 3: Submitting regional cPCA jobs..."

REGIONS=("user" "asst" "special1" "special2")
REGIONAL_JOBS=()

for region in "${REGIONS[@]}"; do
    REGION_FILE="data/activations/regional/conversations2_combined_${region}.h5"
    if [ ! -f "$REGION_FILE" ]; then
        echo "Warning: $REGION_FILE not found, skipping"
        continue
    fi

    # Create config for this region
    CONFIG_FILE="probes/experiments/configs/cpca_conversations_regional_${region}.yaml"
    cat > "$CONFIG_FILE" << EOF
# cPCA configuration for conversation data (regional: ${region})
# Contrasts emotional conversations against neutral paraphrases

name: cpca_conversations_regional_${region}
output_dir: /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_regional_${region}
seed: 42
device: cuda

# Data path (combined emotional + neutral for region ${region})
data_path: /workspace-vast/annas/git/research-tools/data/activations/regional/conversations2_combined_${region}.h5
model_name: google/gemma-3-27b-it

# cPCA parameters
alpha: null  # null = auto-tune per layer
alpha_range: [100, 10000]  # High alpha range based on texts results
n_alphas: 10
n_components: 50
use_diffs: true  # Use diffs (emotional - neutral) as target
EOF

    # Create SLURM script for this region
    SLURM_SCRIPT="probes/scripts/slurm_jobs/run_conversation_cpca_regional_${region}.sh"
    cat > "$SLURM_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=conv_cpca_REGION
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Run regional cPCA on conversation data (REGION)

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Run cPCA experiment
python -m probes.scripts.run_cpca \
    probes/experiments/configs/cpca_conversations_regional_REGION.yaml

echo ""
echo "Regional cPCA (REGION) complete!"
echo "Results in: probes/results/cpca_conversations_regional_REGION"
EOF
    # Replace REGION placeholder
    sed -i "s/REGION/${region}/g" "$SLURM_SCRIPT"
    chmod +x "$SLURM_SCRIPT"

    # Submit job
    JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT")
    REGIONAL_JOBS+=("$JOB_ID")
    echo "  Regional cPCA (${region}): Job $JOB_ID"
done

echo
echo "========================================"
echo "All jobs submitted!"
echo "========================================"
echo "Global: $GLOBAL_JOB"
for i in "${!REGIONS[@]}"; do
    if [ $i -lt ${#REGIONAL_JOBS[@]} ]; then
        echo "Regional (${REGIONS[$i]}): ${REGIONAL_JOBS[$i]}"
    fi
done
echo

echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f /workspace-vast/annas/logs/<job_id>.err"
