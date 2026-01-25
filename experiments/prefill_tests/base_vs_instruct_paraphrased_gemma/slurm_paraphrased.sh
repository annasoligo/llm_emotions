#!/bin/bash
#SBATCH --job-name=gemma_para
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_paraphrased_gemma/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_paraphrased_gemma/logs/%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Gemma paraphrased prefill experiment
# Tests if frustration depends on exact tokens vs semantic content

echo "========================================"
echo "GEMMA PARAPHRASED PREFILL EXPERIMENT"
echo "Models: google/gemma-3-27b-it, google/gemma-3-27b-pt"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

python experiments/base_vs_instruct_paraphrased.py \
    --model-family gemma \
    --phase generate \
    --model-type both \
    --paraphrases-file experiments/base_vs_instruct_paraphrased/shared_paraphrases_20260115_105255.jsonl

# Run judgment after generation
python experiments/base_vs_instruct_paraphrased.py \
    --model-family gemma \
    --phase judge

# Run analysis
python experiments/base_vs_instruct_paraphrased.py \
    --model-family gemma \
    --phase analyze

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "✓ Gemma paraphrased experiment complete"
