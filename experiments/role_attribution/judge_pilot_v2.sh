#!/bin/bash
#SBATCH --job-name=judge_v2_pilot
#SBATCH --output=experiments/role_attribution/logs/judge_v2_pilot_%A_%a.out
#SBATCH --error=experiments/role_attribution/logs/judge_v2_pilot_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-2

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

FILES=(
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_instruct_assistant_20260205_135901.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_instruct_user_20260205_135902.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_instruct_named_20260205_135902.jsonl"
)

INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Processing: $INPUT_FILE"

python experiments/role_attribution/run_judgment.py --input "$INPUT_FILE"
