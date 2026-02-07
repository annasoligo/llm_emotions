#!/bin/bash
#SBATCH --job-name=suppress_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/suppress_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/suppress_qwen235b_%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="vxlan0"

echo "Extracting suppression vectors for Qwen 235B"
echo "============================================="

python steering_tests/suppression_experiments/extract_suppression_vectors.py \
    --model qwen235b \
    --n_samples 100 \
    --seed 42 \
    --output_dir steering_tests/suppression_experiments/vectors \
    --tensor_parallel 4

echo "Done!"
