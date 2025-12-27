#!/bin/bash
# Launch comprehensive multi-seed probe training sweep
# This will train probes at all layers with:
# - Raw activations (n_components=0)
# - 5 PCs (n_components=5)
# - 10 PCs (n_components=10)
# - 20 PCs (n_components=20)
# - 10 seeds per configuration

# Configuration - ALL LAYERS!
LAYERS=($(seq 0 61))  # All 62 layers (0-61)
N_COMPONENTS=(0 5 10 20)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

DATA_PATH="data/activations/texts_combined.h5"
CPCA_PATH="probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz"
OUTPUT_DIR="results/emotion_probes_multiseed"

echo "=========================================="
echo "LAUNCHING MULTI-SEED PROBE TRAINING SWEEP"
echo "=========================================="
echo ""
echo "Layers: ${LAYERS[@]}"
echo "N_components: ${N_COMPONENTS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""
echo "Total jobs: $(( ${#LAYERS[@]} * ${#N_COMPONENTS[@]} * ${#SEEDS[@]} ))"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Launch array job for all configurations
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=probe_sweep
#SBATCH --output=/workspace-vast/annas/logs/probe_sweep_%A_%a.log
#SBATCH --error=/workspace-vast/annas/logs/probe_sweep_%A_%a.err
#SBATCH --array=0-$(( ${#LAYERS[@]} * ${#N_COMPONENTS[@]} * ${#SEEDS[@]} - 1 ))%100
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

set -e

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Change to repo directory
cd /workspace-vast/annas/git/research-tools

# Parse array index into layer, n_components, seed
LAYERS=(${LAYERS[@]})
N_COMPONENTS=(${N_COMPONENTS[@]})
SEEDS=(${SEEDS[@]})

n_layers=\${#LAYERS[@]}
n_ncomps=\${#N_COMPONENTS[@]}
n_seeds=\${#SEEDS[@]}

# Calculate indices
seed_idx=\$(( \$SLURM_ARRAY_TASK_ID % n_seeds ))
temp=\$(( \$SLURM_ARRAY_TASK_ID / n_seeds ))
ncomp_idx=\$(( temp % n_ncomps ))
layer_idx=\$(( temp / n_ncomps ))

layer=\${LAYERS[\$layer_idx]}
n_comp=\${N_COMPONENTS[\$ncomp_idx]}
seed=\${SEEDS[\$seed_idx]}

echo "=========================================="
echo "Job \$SLURM_ARRAY_TASK_ID: Layer=\$layer, n_components=\$n_comp, seed=\$seed"
echo "=========================================="

# Train probe
python probes/scripts/train_emotion_probe_multiseed.py \\
    --data $DATA_PATH \\
    --layer \$layer \\
    --n-components \$n_comp \\
    --seed \$seed \\
    --cpca-results $CPCA_PATH \\
    --output-dir $OUTPUT_DIR

echo ""
echo "Job \$SLURM_ARRAY_TASK_ID completed successfully"
EOF

echo "Array job submitted!"
echo ""
echo "Monitor progress with:"
echo "  squeue -u annas | grep probe_sweep"
echo ""
echo "Check logs in:"
echo "  /workspace-vast/annas/logs/probe_sweep_*.log"
