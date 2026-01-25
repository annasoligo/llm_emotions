#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=baseline_v12
#SBATCH --output=/workspace-vast/annas/logs/baseline_v12_%j.out
#SBATCH --error=/workspace-vast/annas/logs/baseline_v12_%j.err
#SBATCH --time=01:00:00

# Generate baseline responses for V12 solvable prompts
# 6 prompts × 3 responses = 18 total responses

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

cd elicitation

echo "=========================================="
echo "Baseline V12 Response Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

python run_baseline_v12_generation.py

echo ""
echo "=========================================="
echo "Baseline generation complete"
echo "=========================================="

# Show output file
OUTPUT_FILE=$(ls -t outputs/baseline_v12_responses_*.jsonl 2>/dev/null | head -1)
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Output file: $OUTPUT_FILE"
    echo "  Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "  Lines: $(wc -l < "$OUTPUT_FILE")"
else
    echo "✗ No output file found"
    exit 1
fi
