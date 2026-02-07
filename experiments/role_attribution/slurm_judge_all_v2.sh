#!/bin/bash
#SBATCH --job-name=judge_all_v2
#SBATCH --output=experiments/role_attribution/logs/judge_all_v2_%A_%a.out
#SBATCH --error=experiments/role_attribution/logs/judge_all_v2_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --array=0-17

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# All V2 continuation files (sorted alphabetically)
FILES=(
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_base_assistant_20260205_165021.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_base_named_20260205_165022.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_base_user_20260205_165021.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_instruct_assistant_20260205_135901.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_instruct_named_20260205_135902.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_gemma27b_instruct_user_20260205_135902.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_olmo32b_base_assistant_20260205_165021.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_olmo32b_base_named_20260205_165039.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_olmo32b_base_user_20260205_165039.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_olmo32b_instruct_assistant_20260205_165045.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_olmo32b_instruct_named_20260205_165042.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_olmo32b_instruct_user_20260205_165046.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_qwen25b_base_assistant_20260205_165042.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_qwen25b_base_named_20260205_165056.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_qwen25b_base_user_20260205_165056.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_qwen25b_instruct_assistant_20260205_165028.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_qwen25b_instruct_named_20260205_165033.jsonl"
    "experiments/role_attribution/outputs/continuations_v2_qwen25b_instruct_user_20260205_165033.jsonl"
)

INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Processing: $INPUT_FILE"

python experiments/role_attribution/run_judgment.py --input "$INPUT_FILE"
