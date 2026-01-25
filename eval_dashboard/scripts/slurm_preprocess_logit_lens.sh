#!/bin/bash
#SBATCH --job-name=preproc_logit
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/preprocess_logit_lens_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/preprocess_logit_lens_%A.err
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -e

# Load environment
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠ No venv found at .venv/bin/activate"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

# Ensure clean GPU state
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "=============================================================================="
echo "PREPROCESSING DASHBOARD DATA WITH LOGIT LENS"
echo "=============================================================================="
echo ""

# Run preprocessing with logit_lens probes
python eval_dashboard/data_preprocessing.py \
    --input /workspace-vast/annas/git/research-tools/elicitation/outputs/annotated_emotion_onset_gemma3.jsonl \
    --output /workspace-vast/annas/git/research-tools/eval_dashboard/data/preprocessed_with_logit_lens.pkl \
    --probes logit_lens_mean logit_lens_max

echo ""
echo "=============================================================================="
echo "PREPROCESSING COMPLETE"
echo "=============================================================================="
echo ""
echo "Output saved to: eval_dashboard/data/preprocessed_with_logit_lens.pkl"
echo ""
