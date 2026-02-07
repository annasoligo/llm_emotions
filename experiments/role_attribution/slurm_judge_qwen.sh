#!/bin/bash
#SBATCH --job-name=judge_qwen
#SBATCH --output=experiments/role_attribution/logs/judge_qwen_%A_%a.out
#SBATCH --error=experiments/role_attribution/logs/judge_qwen_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --array=0-5

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Array of input files - will be populated after generation completes
FILES=(
    "experiments/role_attribution/outputs/continuations_qwen25b_instruct_assistant_*.jsonl"
    "experiments/role_attribution/outputs/continuations_qwen25b_instruct_user_*.jsonl"
    "experiments/role_attribution/outputs/continuations_qwen25b_instruct_named_*.jsonl"
    "experiments/role_attribution/outputs/continuations_qwen25b_base_assistant_*.jsonl"
    "experiments/role_attribution/outputs/continuations_qwen25b_base_user_*.jsonl"
    "experiments/role_attribution/outputs/continuations_qwen25b_base_named_*.jsonl"
)

# Get file for this array task (expand glob)
INPUT_FILE=$(ls ${FILES[$SLURM_ARRAY_TASK_ID]} | head -1)

echo "Processing: $INPUT_FILE"

python experiments/role_attribution/run_judgment.py --input "$INPUT_FILE"
