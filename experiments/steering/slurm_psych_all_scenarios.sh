#!/bin/bash
#SBATCH --job-name=psych_all
#SBATCH --output=experiments/steering/logs/psych_all_%j.out
#SBATCH --error=experiments/steering/logs/psych_all_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00

# Run all steering experiments on all scenarios (original + additional)
# This script runs: MC, Verbatim, UA, Random experiments

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

# Create log directory
mkdir -p experiments/steering/logs

echo "Starting all psych steering experiments..."
echo "Timestamp: $(date)"

# 1. MC experiment (text mean-diff vectors)
echo ""
echo "=========================================="
echo "1. Running MC experiment (text mean-diff)"
echo "=========================================="
python -m experiments.steering.experiments.psych_decision_steering \
    --mc-only \
    --all-scenarios \
    --norm-pct 0.10

# 2. MC Verbatim experiment
echo ""
echo "=========================================="
echo "2. Running MC Verbatim experiment"
echo "=========================================="
python -m experiments.steering.experiments.psych_decision_steering \
    --mc-verbatim \
    --all-scenarios \
    --norm-pct 0.10

# 3. UA Model experiment
echo ""
echo "=========================================="
echo "3. Running UA Model experiment"
echo "=========================================="
python -m experiments.steering.experiments.psych_decision_steering \
    --ua-mc \
    --all-scenarios \
    --norm-pct 0.10

# 4. Random vector experiment
echo ""
echo "=========================================="
echo "4. Running Random vector experiment"
echo "=========================================="
python -m experiments.steering.experiments.psych_decision_steering \
    --random-mc \
    --all-scenarios \
    --norm-pct 0.10

# 5. Natural language experiment (with judging)
echo ""
echo "=========================================="
echo "5. Running Natural language experiment"
echo "=========================================="
python -m experiments.steering.experiments.psych_decision_steering \
    --natural-only \
    --all-scenarios \
    --norm-pct 0.10 \
    --num-samples 100

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Timestamp: $(date)"
echo "=========================================="
