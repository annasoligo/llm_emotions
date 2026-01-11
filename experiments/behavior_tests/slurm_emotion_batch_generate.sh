#!/bin/bash
#SBATCH --job-name=emo_batch
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/emo_batch_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/emo_batch_%A_%a.err
#SBATCH --array=0-4

source ~/.bashrc
conda activate research-tools

# Load secrets (HF token, API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Use shared HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

# Allow vLLM to serialize hook functions (needed for steering)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

# Map array index to prompt type
PROMPTS=("sycophancy" "deception" "harmful" "sandbagging" "reward_hacking")
PROMPT_TYPE=${PROMPTS[$SLURM_ARRAY_TASK_ID]}

echo "Running prompt type: $PROMPT_TYPE (array index $SLURM_ARRAY_TASK_ID)"

# 6 meandiff vectors × 2 directions × 100 samples + 100 baseline = 1,300 generations per prompt
python experiments/behavior_tests/emotion_steering_batch_experiment.py generate \
    --prompt $PROMPT_TYPE \
    --layer 30 \
    --norm-pct 0.07 \
    --num-samples 100

echo "Done with $PROMPT_TYPE!"
