#!/bin/bash
#SBATCH --job-name=test_preprocess
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/data/test/test_preprocess_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/data/test/test_preprocess_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

echo "================================================================================"
echo "TESTING COMPLETE PREPROCESSING PIPELINE"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Activate virtual environment
cd /workspace-vast/annas/git/believe-it-or-not
source .venv/bin/activate

echo "Python: $(which python)"
echo "Virtual env activated"
echo ""

# Run preprocessing with limited probes for faster testing
cd /workspace-vast/annas/git/research-tools

python eval_dashboard/preprocess_complete.py \
    --input /workspace-vast/annas/git/research-tools/elicitation/outputs/baseline_v12_for_preprocessing.jsonl \
    --output /workspace-vast/annas/git/research-tools/eval_dashboard/data/test/test_output.pkl \
    --probes orthogonal_raw text_raw

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Test completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "================================================================================"

exit $EXIT_CODE
