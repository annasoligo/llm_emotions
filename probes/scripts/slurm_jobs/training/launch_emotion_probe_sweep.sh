#!/bin/bash
# Launch emotion probe training sweep with automatic visualization
# Usage: ./probes/scripts/slurm_jobs/launch_emotion_probe_sweep.sh

cd /workspace-vast/annas/git/research-tools || exit 1

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

# Layers to train on (comma-separated, no spaces)
LAYERS="10,20,30,40,50"

# Number of PCs to test (space-separated)
N_COMPONENTS_LIST="3 5 10 20 50"

# Data and cPCA paths
DATA_PATH="data/activations/texts_combined.h5"
CPCA_RESULTS="probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz"

# Training hyperparameters (usually don't need to change these)
WEIGHT_DECAY="1.0"      # L2 regularization (1.0 recommended)
L1_LAMBDA="0.0"         # L1 regularization (0.0 = disabled)
MAX_SAMPLES=""          # Auto-balance classes (leave empty)
LEARNING_RATE="0.001"
BATCH_SIZE="64"
MAX_EPOCHS="500"
PATIENCE="10"

# Output directory prefix
OUTPUT_DIR_PREFIX="results/emotion_probes"

# ============================================================================
# LAUNCH JOBS
# ============================================================================

echo "=========================================="
echo "Launching Emotion Probe Training Sweep"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Layers: $LAYERS"
echo "  N Components: $N_COMPONENTS_LIST"
echo "  Data: $DATA_PATH"
echo "  cPCA: $CPCA_RESULTS"
echo "  Weight decay (L2): $WEIGHT_DECAY"
echo "  L1 lambda: $L1_LAMBDA"
echo ""

# Create log directory
mkdir -p slurm_logs

# Track job IDs for later
declare -a JOB_IDS

# Launch jobs for each N_COMPONENTS setting
for n_pcs in $N_COMPONENTS_LIST; do
    if [ "$n_pcs" = "50" ]; then
        # Use standard output directory for baseline (50 PCs)
        output_dir="${OUTPUT_DIR_PREFIX}_high_alpha_cpca"
    else
        # Use top{N} suffix for reduced dimensionality
        output_dir="${OUTPUT_DIR_PREFIX}_top${n_pcs}"
    fi

    echo "=== Launching: Top $n_pcs PCs ==="
    echo "  Output: $output_dir"

    # Set environment variables and launch
    if [ "$n_pcs" = "50" ]; then
        # Don't set N_COMPONENTS for 50 (use all)
        job_output=$(DATA_PATH="$DATA_PATH" \
                    USE_CPCA=1 \
                    CPCA_RESULTS="$CPCA_RESULTS" \
                    OUTPUT_DIR="$output_dir" \
                    WEIGHT_DECAY="$WEIGHT_DECAY" \
                    L1_LAMBDA="$L1_LAMBDA" \
                    MAX_SAMPLES="$MAX_SAMPLES" \
                    LEARNING_RATE="$LEARNING_RATE" \
                    BATCH_SIZE="$BATCH_SIZE" \
                    MAX_EPOCHS="$MAX_EPOCHS" \
                    PATIENCE="$PATIENCE" \
                    sbatch --array="$LAYERS" probes/scripts/slurm_jobs/train_emotion_probe.sh)
    else
        # Set N_COMPONENTS for reduced dimensionality
        job_output=$(N_COMPONENTS="$n_pcs" \
                    DATA_PATH="$DATA_PATH" \
                    USE_CPCA=1 \
                    CPCA_RESULTS="$CPCA_RESULTS" \
                    OUTPUT_DIR="$output_dir" \
                    WEIGHT_DECAY="$WEIGHT_DECAY" \
                    L1_LAMBDA="$L1_LAMBDA" \
                    MAX_SAMPLES="$MAX_SAMPLES" \
                    LEARNING_RATE="$LEARNING_RATE" \
                    BATCH_SIZE="$BATCH_SIZE" \
                    MAX_EPOCHS="$MAX_EPOCHS" \
                    PATIENCE="$PATIENCE" \
                    sbatch --array="$LAYERS" probes/scripts/slurm_jobs/train_emotion_probe.sh)
    fi

    # Extract job ID
    job_id=$(echo $job_output | grep -oP '\d+')
    JOB_IDS+=($job_id)
    echo "  Job ID: $job_id"
    echo ""
done

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "To monitor progress:"
echo "  squeue -u $USER | grep emotion"
echo ""
echo "Once complete, visualizations will be automatically generated:"
echo "  - Dimensionality sweep plot"
echo "  - Weight distribution analysis"
echo "  - Summary tables"
echo ""
echo "Waiting for jobs to complete..."

# ============================================================================
# WAIT FOR COMPLETION AND GENERATE VISUALIZATIONS
# ============================================================================

# Function to check if all jobs are done
all_jobs_done() {
    for job_id in "${JOB_IDS[@]}"; do
        if squeue -j $job_id 2>/dev/null | grep -q $job_id; then
            return 1  # Job still running
        fi
    done
    return 0  # All done
}

# Wait for completion (check every 30 seconds)
while ! all_jobs_done; do
    sleep 30
done

echo ""
echo "=========================================="
echo "All jobs complete! Generating visualizations..."
echo "=========================================="
echo ""

# Generate all visualizations
python probes/scripts/generate_all_visualizations.py
