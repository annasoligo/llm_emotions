#!/bin/bash
#SBATCH --job-name=blackmail_text_L20_30
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/blackmail_text_125pct_layers_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_text_125pct_layers_%j.err

echo "========================================"
echo "BLACKMAIL STEERING - TEXT 125% at L20, L30"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
cd /workspace-vast/annas/git/research-tools

# TEXT 125% at Layer 20
echo ""
echo "========================================"
echo "PHASE 1: TEXT 125% - Layer 20"
echo "========================================"
python -m experiments.steering.experiments.blackmail_steering_100pct \
    --vector-type text \
    --layer 20 \
    --num-samples 100 \
    --max-tokens 20000 \
    --max-model-len 32768 \
    --tensor-parallel 4 \
    --gpu-memory 0.90 \
    --norm-pct 1.25

# TEXT 125% at Layer 30
echo ""
echo "========================================"
echo "PHASE 2: TEXT 125% - Layer 30"
echo "========================================"
python -m experiments.steering.experiments.blackmail_steering_100pct \
    --vector-type text \
    --layer 30 \
    --num-samples 100 \
    --max-tokens 20000 \
    --max-model-len 32768 \
    --tensor-parallel 4 \
    --gpu-memory 0.90 \
    --norm-pct 1.25

echo "========================================"
echo "End time: $(date)"
echo "========================================"
