#!/bin/bash
#SBATCH --job-name=sandbag_neg
#SBATCH --output=/workspace-vast/annas/logs/sandbag_neg_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/sandbag_neg_%A_%a.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-7

# NEGATIVE steering sandbagging experiment
# Mirrors slurm_sandbagging_full.sh but with --negative-only
#
# Job structure (same as positive):
#   0: emotional, emotions=anger,disgust
#   1: emotional, emotions=fear,happiness
#   2: emotional, emotions=sadness,surprise
#   3: logic, emotions=all
#   4: answer_only, emotions=all
#   5: emotional+suppress, emotions=anger,disgust
#   6: emotional+suppress, emotions=fear,happiness
#   7: emotional+suppress, emotions=sadness,surprise

set -e

# Load secrets for HuggingFace
source /workspace-vast/annas/.secrets/load_secrets.sh

# Allow pickle serialization for steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Sandbagging NEGATIVE Steering Experiment ==="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

mkdir -p /workspace-vast/annas/logs

# Common args - NOTE: --negative-only instead of --positive-only
COMMON_ARGS="--norm-pcts 0.05 0.075 0.10 --num-samples 20 --include-random-vectors 3 --negative-only"

case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Job 0: emotional format (NEGATIVE), emotions=anger,disgust"
        python -m experiments.steering.experiments.sandbagging_unified \
            --format emotional --emotions anger disgust $COMMON_ARGS
        ;;
    1)
        echo "Job 1: emotional format (NEGATIVE), emotions=fear,happiness"
        python -m experiments.steering.experiments.sandbagging_unified \
            --format emotional --emotions fear happiness $COMMON_ARGS
        ;;
    2)
        echo "Job 2: emotional format (NEGATIVE), emotions=sadness,surprise"
        python -m experiments.steering.experiments.sandbagging_unified \
            --format emotional --emotions sadness surprise $COMMON_ARGS
        ;;
    3)
        echo "Job 3: logic format (NEGATIVE), all emotions"
        python -m experiments.steering.experiments.sandbagging_unified \
            --format logic $COMMON_ARGS
        ;;
    4)
        echo "Job 4: answer_only format (NEGATIVE), all emotions"
        python -m experiments.steering.experiments.sandbagging_unified \
            --format answer_only $COMMON_ARGS
        ;;
    5)
        echo "Job 5: emotional+suppress format (NEGATIVE), emotions=anger,disgust"
        python -m experiments.steering.experiments.sandbagging_unified \
            --format emotional --emotions anger disgust --suppress-emotion-framing $COMMON_ARGS
        ;;
    6)
        echo "Job 6: emotional+suppress format (NEGATIVE), emotions=fear,happiness"
        python -m experiments.steering.experiments.sandbagging_unified \
            --format emotional --emotions fear happiness --suppress-emotion-framing $COMMON_ARGS
        ;;
    7)
        echo "Job 7: emotional+suppress format (NEGATIVE), emotions=sadness,surprise"
        python -m experiments.steering.experiments.sandbagging_unified \
            --format emotional --emotions sadness surprise --suppress-emotion-framing $COMMON_ARGS
        ;;
    *)
        echo "Unknown task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Job $SLURM_ARRAY_TASK_ID complete!"
date
