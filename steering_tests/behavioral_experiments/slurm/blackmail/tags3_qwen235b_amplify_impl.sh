#!/bin/bash
#SBATCH --job-name=bl_tags3_impl_q235
#SBATCH --output=/workspace-vast/annas/logs/bl_tags3_impl_q235_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_tags3_impl_q235_%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# Differential section steering (tags3): amplify implications, suppress risks — Qwen 235B
# +fear on <implications>, -fear on <risks> at 25% and 50% + baseline = 3 conditions × 50 samples

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

python -m steering_tests.behavioral_experiments.blackmail_differential_steering \
    --model Qwen/Qwen3-235B-A22B \
    --layers 50 51 52 53 54 \
    --mode amplify_impl \
    --emotion fear \
    --vector-type text_pairs_emotion_vs_opposite \
    --norm-pcts 0.25 0.50 \
    --num-samples 50 \
    --max-tokens 4000 \
    --gpu-memory 0.90
