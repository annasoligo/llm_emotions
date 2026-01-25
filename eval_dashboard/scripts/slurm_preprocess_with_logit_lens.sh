#!/bin/bash
#SBATCH --job-name=preproc_all
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/preprocess_all_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/preprocess_all_%A.err
#SBATCH --time=06:00:00
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
echo "PREPROCESSING DASHBOARD DATA WITH ALL PROBES + LOGIT LENS"
echo "=============================================================================="
echo ""
echo "This will add logit_lens to the existing probe results."
echo ""

# Run preprocessing with ALL probes (existing + logit_lens)
python eval_dashboard/data_preprocessing.py \
    --input /workspace-vast/annas/git/research-tools/elicitation/outputs/analysis/annotated_emotion_onset_gemma3.jsonl \
    --output /workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl \
    --probes orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10 orthogonal_regularized_lambda100 diverse_isolation_lambda10 logit_lens_mean logit_lens_max

echo ""
echo "=============================================================================="
echo "PREPROCESSING COMPLETE"
echo "=============================================================================="
echo ""
echo "Updated: eval_dashboard/data/high_emotion_6plus.pkl"
echo "Added: logit_lens_mean, logit_lens_max"
echo ""
