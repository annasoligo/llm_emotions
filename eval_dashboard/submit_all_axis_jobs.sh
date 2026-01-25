#!/bin/bash
# Submit SLURM jobs to process all eval_dashboard datasets with axis scores

DATA_DIR="/workspace-vast/annas/git/research-tools/eval_dashboard/data"
SCRIPT_DIR="/workspace-vast/annas/git/research-tools/eval_dashboard"

# List of datasets to process
DATASETS=(
    "mid_emotion_3to5.pkl"
    "low_emotion_0to2.pkl"
    "low_emotion_no_shutdown.pkl"
    "low_emotion_with_shutdown.pkl"
    "baseline_v12_solvable.pkl"
)

echo "========================================================================"
echo "Submitting Axis Preprocessing Jobs for All Datasets"
echo "========================================================================"

for dataset in "${DATASETS[@]}"; do
    input_path="$DATA_DIR/$dataset"
    output_name="${dataset/.pkl/_with_axes.pkl}"
    output_path="$DATA_DIR/$output_name"

    # Check if input exists
    if [ ! -f "$input_path" ]; then
        echo "SKIP: $dataset (not found)"
        continue
    fi

    # Check if already processed
    if [ -f "$output_path" ]; then
        echo "SKIP: $dataset (already has axes)"
        continue
    fi

    echo ""
    echo "Submitting: $dataset"
    echo "  Input:  $input_path"
    echo "  Output: $output_path"

    # Create SLURM script for this dataset
    job_name="axis_${dataset/.pkl/}"
    log_file="/workspace-vast/annas/logs/${job_name}_%j.log"
    err_file="/workspace-vast/annas/logs/${job_name}_%j.err"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=$log_file
#SBATCH --error=$err_file
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "Job started at \$(date)"
echo "Processing: $dataset"

# Change to project directory
cd $SCRIPT_DIR

# Load secrets
if [ -f /workspace-vast/annas/.secrets ]; then
    source /workspace-vast/annas/.secrets
fi

# Activate venv
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

# Run preprocessing with arguments
python3 add_axis_lens_to_existing.py "$input_path" "$output_path"

echo "Job completed at \$(date)"
EOF

done

echo ""
echo "========================================================================"
echo "All jobs submitted! Check status with: squeue -u \$USER"
echo "========================================================================"
