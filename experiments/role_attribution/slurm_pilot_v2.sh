#!/bin/bash
#SBATCH --job-name=pilot_v2
#SBATCH --output=experiments/role_attribution/logs/pilot_v2_%j.out
#SBATCH --error=experiments/role_attribution/logs/pilot_v2_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --array=0-2

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Test with Gemma-27B instruct only, all 3 roles
ROLES=(assistant user named)
ROLE=${ROLES[$SLURM_ARRAY_TASK_ID]}

echo "Pilot V2: Gemma-27B instruct, role=$ROLE"

python experiments/role_attribution/run_generation_v2.py \
    --model-family gemma27b \
    --model-type instruct \
    --data experiments/role_attribution/outputs/prepared_statements_v2_20260205_135808.json \
    --role $ROLE
