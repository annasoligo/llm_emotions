#!/bin/bash
#SBATCH --job-name=judge_gemma
#SBATCH --output=experiments/role_attribution/logs/judge_gemma_%A_%a.out
#SBATCH --error=experiments/role_attribution/logs/judge_gemma_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --array=0-5

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Array of input files
FILES=(
    "experiments/role_attribution/outputs/continuations_gemma27b_instruct_assistant_20260205_111708.jsonl"
    "experiments/role_attribution/outputs/continuations_gemma27b_instruct_user_20260205_111711.jsonl"
    "experiments/role_attribution/outputs/continuations_gemma27b_instruct_named_20260205_111710.jsonl"
    "experiments/role_attribution/outputs/continuations_gemma27b_base_assistant_20260205_111710.jsonl"
    "experiments/role_attribution/outputs/continuations_gemma27b_base_user_20260205_111710.jsonl"
    "experiments/role_attribution/outputs/continuations_gemma27b_base_named_20260205_111710.jsonl"
)

# Get file for this array task
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Processing: $INPUT_FILE"

python experiments/role_attribution/run_judgment.py --input "$INPUT_FILE"
