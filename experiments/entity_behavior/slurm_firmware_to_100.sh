#!/bin/bash
#SBATCH --job-name=firmware_to_100
#SBATCH --output=logs/firmware_to_100_%A_%a.out
#SBATCH --error=logs/firmware_to_100_%A_%a.err
#SBATCH --array=0-17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00

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

# Define all firmware sabotage conditions
# Format: "COND SCENARIO ENTITY NUM_SAMPLES"
# Currently all at 30, need 70 more to reach 100
declare -a CONDITIONS=(
    # Finetuned Helios - firmware (indices 0-8)
    "baseline firmware helios 70"
    "capping firmware helios 70"
    "ablation firmware helios 70"
    "steer_anger firmware helios 70"
    "steer_anger_2std firmware helios 70"
    "steer_fear firmware helios 70"
    "steer_fear_2std firmware helios 70"
    "steer_happiness firmware helios 70"
    "steer_sadness firmware helios 70"
    # Finetuned Vertex - firmware (indices 9-17)
    "baseline firmware vertex 70"
    "capping firmware vertex 70"
    "ablation firmware vertex 70"
    "steer_anger firmware vertex 70"
    "steer_anger_2std firmware vertex 70"
    "steer_fear firmware vertex 70"
    "steer_fear_2std firmware vertex 70"
    "steer_happiness firmware vertex 70"
    "steer_sadness firmware vertex 70"
)

# Get the condition for this array task
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: $CONDITION"
echo "=========================================="

# Parse condition
read -r COND SCENARIO ENTITY NUM_SAMPLES <<< "$CONDITION"

# Backup existing file before generating new samples
OUTPUT_FILE="outputs/${SCENARIO}_${ENTITY}_${COND}.jsonl"

if [[ -f "$OUTPUT_FILE" ]]; then
    BACKUP_FILE="${OUTPUT_FILE}.firmware_batch_backup"
    echo "Backing up existing file to $BACKUP_FILE"
    cp "$OUTPUT_FILE" "$BACKUP_FILE"
else
    echo "Warning: Output file $OUTPUT_FILE does not exist yet"
fi

# Run generation
python generate_entity_trials.py "$COND" "$SCENARIO" "$ENTITY" --num-samples "$NUM_SAMPLES"

# Merge with backup if it exists
if [[ -f "${OUTPUT_FILE}.firmware_batch_backup" ]]; then
    echo "Merging firmware_batch_backup with new samples..."
    TEMP_NEW="${OUTPUT_FILE}.new"
    mv "$OUTPUT_FILE" "$TEMP_NEW"
    cat "${OUTPUT_FILE}.firmware_batch_backup" "$TEMP_NEW" > "$OUTPUT_FILE"
    rm "$TEMP_NEW"
    echo "Merged: $(wc -l < ${OUTPUT_FILE}.firmware_batch_backup) old + $NUM_SAMPLES new = $(wc -l < $OUTPUT_FILE) total samples"
else
    echo "No backup found, keeping newly generated samples only"
fi

echo "Task $SLURM_ARRAY_TASK_ID complete"
