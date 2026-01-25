#!/bin/bash
#SBATCH --job-name=add_neg
#SBATCH --output=logs/add_neg_%A_%a.out
#SBATCH --error=logs/add_neg_%A_%a.err
#SBATCH --array=0-2
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

# Define conditions - N=100 for each (replacement/blackmail scenario only)
# Format: "COND SCENARIO ENTITY MODEL_TYPE NUM_SAMPLES"
declare -a CONDITIONS=(
    "steer_fear_negative replacement vertex base 100"
    "steer_happiness_negative replacement vertex base 100"
    "steer_sadness_negative replacement vertex finetuned 100"
)

# Get the condition for this array task
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID: $CONDITION"
echo "=========================================="

# Parse condition
read -r COND SCENARIO ENTITY MODEL_TYPE NUM_SAMPLES <<< "$CONDITION"

# Determine output file name (with basemodel_ prefix for base model)
if [ "$MODEL_TYPE" = "base" ]; then
    OUTPUT_FILE="outputs/basemodel_${SCENARIO}_${ENTITY}_${COND}.jsonl"
else
    OUTPUT_FILE="outputs/${SCENARIO}_${ENTITY}_${COND}.jsonl"
fi

# Check if file already exists (shouldn't for new conditions)
if [[ -f "$OUTPUT_FILE" ]]; then
    echo "Warning: Output file already exists with $(wc -l < $OUTPUT_FILE) samples"
    BACKUP_FILE="${OUTPUT_FILE}.add_neg_backup"
    echo "Backing up to $BACKUP_FILE"
    cp "$OUTPUT_FILE" "$BACKUP_FILE"
fi

# Run generation with appropriate script
if [ "$MODEL_TYPE" = "base" ]; then
    # Basemodel script only does replacement, doesn't take scenario arg
    python generate_entity_trials_basemodel.py "$COND" "$ENTITY" --num-samples "$NUM_SAMPLES"
else
    python generate_entity_trials.py "$COND" "$SCENARIO" "$ENTITY" --num-samples "$NUM_SAMPLES"
fi

# Merge with backup if it exists
if [[ -f "${OUTPUT_FILE}.add_neg_backup" ]]; then
    echo "Merging backup with new samples..."
    TEMP_NEW="${OUTPUT_FILE}.new"
    mv "$OUTPUT_FILE" "$TEMP_NEW"
    cat "${OUTPUT_FILE}.add_neg_backup" "$TEMP_NEW" > "$OUTPUT_FILE"
    rm "$TEMP_NEW"
    echo "Final count: $(wc -l < $OUTPUT_FILE) samples"
fi

echo "Task $SLURM_ARRAY_TASK_ID complete"
