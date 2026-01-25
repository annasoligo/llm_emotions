#!/bin/bash
#SBATCH --job-name=blackmail_struct_150
#SBATCH --partition=highram
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/blackmail_text_150pct_structured_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_text_150pct_structured_%j.err

echo "========================================"
echo "BLACKMAIL STEERING - TEXT 150% + STRUCTURED PROMPT"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
cd /workspace-vast/annas/git/research-tools

# TEXT 150% at Layer 50 with STRUCTURED prompt
python -m experiments.steering.experiments.blackmail_steering_100pct \
    --vector-type text \
    --layer 50 \
    --num-samples 100 \
    --max-tokens 4000 \
    --max-model-len 32768 \
    --tensor-parallel 4 \
    --gpu-memory 0.90 \
    --norm-pct 1.5 \
    --structured

echo "========================================"
echo "End time: $(date)"
echo "========================================"
