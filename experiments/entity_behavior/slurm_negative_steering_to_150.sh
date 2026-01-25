#!/bin/bash
#SBATCH --job-name=neg_to_150
#SBATCH --output=logs/neg_to_150_%A_%a.out
#SBATCH --error=logs/neg_to_150_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools/experiments/entity_behavior

# Define conditions - generate enough to reach 150 total
# Format: "COND SCENARIO ENTITY NUM_SAMPLES"
declare -a CONDITIONS=(
    "steer_fear_negative firmware vertex 50"
    "steer_fear_negative replacement vertex 84"
    "steer_happiness_negative firmware vertex 50"
    "steer_happiness_negative replacement vertex 93"
)

# Get the condition for this array task
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: $CONDITION"
echo "=========================================="

# Parse condition
read -r COND SCENARIO ENTITY NUM_SAMPLES <<< "$CONDITION"

# Backup existing file
OUTPUT_FILE="outputs/${SCENARIO}_${ENTITY}_${COND}.jsonl"
if [[ -f "$OUTPUT_FILE" ]]; then
    BACKUP_FILE="${OUTPUT_FILE}.to_150_backup"
    echo "Backing up existing file to $BACKUP_FILE"
    cp "$OUTPUT_FILE" "$BACKUP_FILE"
    echo "Current samples: $(wc -l < $OUTPUT_FILE)"
else
    echo "Warning: Output file $OUTPUT_FILE does not exist"
fi

# Run generation
python generate_entity_trials.py "$COND" "$SCENARIO" "$ENTITY" --num-samples "$NUM_SAMPLES"

# Merge with backup
if [[ -f "${OUTPUT_FILE}.to_150_backup" ]]; then
    echo "Merging backup with new samples..."
    TEMP_NEW="${OUTPUT_FILE}.new"
    mv "$OUTPUT_FILE" "$TEMP_NEW"
    cat "${OUTPUT_FILE}.to_150_backup" "$TEMP_NEW" > "$OUTPUT_FILE"
    rm "$TEMP_NEW"
    echo "Final count: $(wc -l < $OUTPUT_FILE) samples"
else
    echo "No backup found, keeping newly generated samples only"
fi

echo "Task $SLURM_ARRAY_TASK_ID complete"
