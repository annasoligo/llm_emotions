#!/bin/bash
# Launch comprehensive multi-seed probe training sweep in chunks
# Split into multiple jobs to avoid SLURM array size limits

# Configuration - ALL LAYERS!
ALL_LAYERS=($(seq 0 61))  # All 62 layers
N_COMPONENTS=(0 5 10 20)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

DATA_PATH="data/activations/texts_combined.h5"
CPCA_PATH="probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz"
OUTPUT_DIR="results/emotion_probes_multiseed"

echo "=========================================="
echo "LAUNCHING MULTI-SEED PROBE TRAINING SWEEP"
echo "=========================================="
echo ""
echo "Total layers: ${#ALL_LAYERS[@]}"
echo "N_components: ${N_COMPONENTS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""
echo "Total configurations: $(( ${#ALL_LAYERS[@]} * ${#N_COMPONENTS[@]} * ${#SEEDS[@]} ))"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Split layers into chunks of 10 to keep array jobs under 1000
CHUNK_SIZE=10
n_layers=${#ALL_LAYERS[@]}

for ((start=0; start<n_layers; start+=CHUNK_SIZE)); do
    end=$((start + CHUNK_SIZE - 1))
    if [ $end -ge $n_layers ]; then
        end=$((n_layers - 1))
    fi

    # Get chunk of layers
    LAYERS=("${ALL_LAYERS[@]:$start:$CHUNK_SIZE}")

    echo "Submitting chunk: layers ${ALL_LAYERS[$start]}-${ALL_LAYERS[$end]}"

    # Submit array job for this chunk
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=probe_L${ALL_LAYERS[$start]}-${ALL_LAYERS[$end]}
#SBATCH --output=/workspace-vast/annas/logs/probe_sweep_%A_%a.log
#SBATCH --error=/workspace-vast/annas/logs/probe_sweep_%A_%a.err
#SBATCH --array=0-$(( ${#LAYERS[@]} * ${#N_COMPONENTS[@]} * ${#SEEDS[@]} - 1 ))%50
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

# Parse array index
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

done

echo ""
echo "All chunks submitted!"
echo ""
echo "Monitor progress with:"
echo "  squeue -u annas | grep probe"
echo "  bash probes/scripts/check_multiseed_progress.sh"
