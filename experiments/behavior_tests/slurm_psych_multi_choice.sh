#!/bin/bash
#SBATCH --job-name=psych_multi
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/behavior_tests/logs/psych_multi_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/behavior_tests/logs/psych_multi_%j.err
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

mkdir -p experiments/behavior_tests/logs

# Shared run ID for all models
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/behavior_tests/outputs/psych_multi_choice/${RUN_ID}"

echo "Run ID: ${RUN_ID}"
echo "Output dir: ${OUTPUT_DIR}"

# Run all three models sequentially with same run ID
# First Gemma (smaller, fits easily)
echo "Running Gemma-3-27B..."
python -m experiments.behavior_tests.psych_multi_choice_experiment \
    --models gemma \
    --run-id ${RUN_ID}

# Then Llama 70B (larger, needs more memory)
echo "Running Llama-3.3-70B..."
python -m experiments.behavior_tests.psych_multi_choice_experiment \
    --models llama \
    --run-id ${RUN_ID}

# Finally GPT-5-mini via API (no GPU needed)
echo "Running GPT-5-mini..."
python -m experiments.behavior_tests.psych_multi_choice_experiment \
    --models gpt5 \
    --run-id ${RUN_ID}

# Generate combined plots
echo "Generating combined plots..."
python -m experiments.behavior_tests.psych_multi_choice_experiment \
    --combine-results ${OUTPUT_DIR}

echo "Done! Results in: ${OUTPUT_DIR}"
