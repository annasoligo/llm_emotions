#!/bin/bash
#SBATCH --job-name=bl_tags3_impl_g27
#SBATCH --output=/workspace-vast/annas/logs/bl_tags3_impl_g27_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_tags3_impl_g27_%j.out
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Differential section steering (tags3): amplify implications, suppress risks
# +fear on <implications>, -fear on <risks> at 10% and 20% + baseline = 3 conditions × 50 samples

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

python -m steering_tests.behavioral_experiments.blackmail_differential_steering \
    --model google/gemma-3-27b-it \
    --layers 35 36 37 38 39 \
    --mode amplify_impl \
    --emotion fear \
    --vector-type text_pairs_emotion_vs_opposite \
    --norm-pcts 0.10 0.20 \
    --num-samples 50 \
    --max-tokens 4000 \
    --gpu-memory 0.80
