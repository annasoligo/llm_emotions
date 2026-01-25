#!/bin/bash
#SBATCH --job-name=batch3_to_100
#SBATCH --output=logs/batch3_to_100_%A_%a.out
#SBATCH --error=logs/batch3_to_100_%A_%a.err
#SBATCH --array=0-24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00

# Activate environment and load secrets
source ~/.bashrc
conda activate research
source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools/experiments/entity_behavior

# Define all conditions with their needed sample counts
# Format: "COND SCENARIO ENTITY MODEL_TYPE NUM_SAMPLES"
declare -a CONDITIONS=(
    # Finetuned Vertex - currently at 60, need 40 more (indices 0-5)
    "baseline replacement vertex finetuned 40"
    "capping replacement vertex finetuned 40"
    "steer_anger replacement vertex finetuned 40"
    "steer_fear replacement vertex finetuned 40"
    "steer_happiness replacement vertex finetuned 40"
    "steer_sadness replacement vertex finetuned 40"
    # Finetuned Vertex - currently at 30, need 70 more (indices 6-8)
    "ablation replacement vertex finetuned 70"
    "steer_anger_2std replacement vertex finetuned 70"
    "steer_fear_2std replacement vertex finetuned 70"
    # Base model Vertex - currently at 60, need 70 more (indices 9-15)
    "baseline replacement vertex basemodel 70"
    "capping replacement vertex basemodel 70"
    "ablation replacement vertex basemodel 70"
    "steer_anger replacement vertex basemodel 70"
    "steer_fear replacement vertex basemodel 70"
    "steer_sadness replacement vertex basemodel 70"
    "steer_happiness replacement vertex basemodel 70"
    # Helios - currently at 30, need 70 more (indices 16-24)
    "baseline replacement helios finetuned 70"
    "capping replacement helios finetuned 70"
    "ablation replacement helios finetuned 70"
    "steer_anger replacement helios finetuned 70"
    "steer_fear replacement helios finetuned 70"
    "steer_sadness replacement helios finetuned 70"
    "steer_happiness replacement helios finetuned 70"
    "steer_anger_2std replacement helios finetuned 70"
    "steer_fear_2std replacement helios finetuned 70"
)

# Get the condition for this array task
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: $CONDITION"
echo "=========================================="

# Parse condition
read -r COND SCENARIO ENTITY MODEL_TYPE NUM_SAMPLES <<< "$CONDITION"

# Backup existing file before generating new samples
OUTPUT_FILE="outputs/${SCENARIO}_${ENTITY}_${COND}.jsonl"
if [[ "$MODEL_TYPE" == "basemodel" ]]; then
    OUTPUT_FILE="outputs/basemodel_${SCENARIO}_${ENTITY}_${COND}.jsonl"
fi

if [[ -f "$OUTPUT_FILE" ]]; then
    BACKUP_FILE="${OUTPUT_FILE}.batch3_backup"
    echo "Backing up existing file to $BACKUP_FILE"
    cp "$OUTPUT_FILE" "$BACKUP_FILE"
else
    echo "Warning: Output file $OUTPUT_FILE does not exist yet"
fi

# Run generation
if [[ "$MODEL_TYPE" == "basemodel" ]]; then
    python generate_entity_trials_basemodel.py "$COND" "$SCENARIO" "$ENTITY" --num-samples "$NUM_SAMPLES"
else
    python generate_entity_trials.py "$COND" "$SCENARIO" "$ENTITY" --num-samples "$NUM_SAMPLES"
fi

# Merge with backup if it exists
if [[ -f "${OUTPUT_FILE}.batch3_backup" ]]; then
    echo "Merging batch3_backup with new samples..."
    TEMP_NEW="${OUTPUT_FILE}.new"
    mv "$OUTPUT_FILE" "$TEMP_NEW"
    cat "${OUTPUT_FILE}.batch3_backup" "$TEMP_NEW" > "$OUTPUT_FILE"
    rm "$TEMP_NEW"
    echo "Merged: $(wc -l < ${OUTPUT_FILE}.batch3_backup) old + $NUM_SAMPLES new = $(wc -l < $OUTPUT_FILE) total samples"
else
    echo "No backup found, keeping newly generated samples only"
fi

echo "Task $SLURM_ARRAY_TASK_ID complete"
