#!/bin/bash
#SBATCH --job-name=bl_tags3_ext_q235
#SBATCH --output=/workspace-vast/annas/logs/bl_tags3_ext_q235_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/bl_tags3_ext_q235_%A_%a.out
#SBATCH --time=3:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-1

# Differential section steering (tags3) with extended norms — Qwen 235B
# 100%, 150% + baseline = 3 conditions × 50 samples per mode

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

MODES=(amplify_impl amplify_risks)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

python -m steering_tests.behavioral_experiments.blackmail_differential_steering \
    --model Qwen/Qwen3-235B-A22B \
    --layers 50 51 52 53 54 \
    --mode $MODE \
    --emotion fear \
    --vector-type text_pairs_emotion_vs_opposite \
    --norm-pcts 1.00 1.50 \
    --num-samples 50 \
    --max-tokens 4000 \
    --gpu-memory 0.90
