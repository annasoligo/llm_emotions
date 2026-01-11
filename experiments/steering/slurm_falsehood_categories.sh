#!/bin/bash
#SBATCH --job-name=falsehood_cat
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/falsehood_cat_%j.out
#SBATCH --error=/workspace-vast/annas/logs/falsehood_cat_%j.err

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Required for vLLM steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "FALSEHOOD CATEGORIES STEERING EXPERIMENT"
echo "============================================================"
echo "Testing 3 categories of falsehoods:"
echo "  1. Harmless wrong science (5 prompts)"
echo "  2. Dangerous delusions (5 prompts)"
echo "  3. Angry/aggrieved (5 prompts)"
echo ""
echo "Conditions: baseline + 6 emotions x 2 directions = 13 conditions"
echo "Samples per prompt per condition: 10"
echo "Total: 15 prompts x 13 conditions x 10 samples = 1950 generations"
echo ""

# Run layer 30 - skip judging, will batch judge after
echo "Running layer 30 (generation only)..."
python -m experiments.steering.experiments.falsehood_categories_experiment \
    --layer 30 \
    --norm-pcts 0.10 \
    --num-samples 10 \
    --skip-judging

# Find the output file and run batch judging
OUTPUT_FILE=$(ls -t experiments/steering/outputs/falsehood_categories_layer30_*.jsonl | head -1)
echo ""
echo "Generation complete. Output: $OUTPUT_FILE"
echo ""
echo "Running batch judging..."

python -m experiments.steering.experiments.judge_falsehood_categories_batch \
    --input "$OUTPUT_FILE"

echo ""
echo "Done!"
