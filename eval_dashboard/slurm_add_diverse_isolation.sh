#!/bin/bash
#SBATCH --job-name=add_diverse_iso
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

cd /workspace-vast/annas/git/research-tools/eval_dashboard

source /workspace-vast/annas/git/research-tools/.venv/bin/activate

echo "Starting diverse isolation probe preprocessing..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Timestamp: $(date)"

python3 add_diverse_isolation_probes.py --subset all

echo "Completed at: $(date)"
