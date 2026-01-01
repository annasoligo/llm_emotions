#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=preprocess_dashboard
#SBATCH --output=/workspace-vast/annas/logs/preprocess_dashboard_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/preprocess_dashboard_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --array=0-4

# SLURM Array Job for Dashboard Subset Preprocessing
# Each array task processes one subset with all 5 probes
#
# Usage: sbatch slurm_preprocess_subset.sh

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

cd /workspace-vast/annas/git/research-tools/eval_dashboard

PROBES="orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10"

# Define subsets array
SUBSETS=(
    "high_emotion_6plus"
    "mid_emotion_3to5"
    "low_emotion_0to2"
    "low_emotion_no_shutdown"
    "low_emotion_with_shutdown"
)

SUBSET=${SUBSETS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Dashboard Preprocessing - Array Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Subset: $SUBSET"
echo "Probes: $PROBES"
echo "=========================================="

python data_preprocessing.py \
  --input ../elicitation/outputs/dashboard_subsets/${SUBSET}.jsonl \
  --output data/${SUBSET}.pkl \
  --probes $PROBES

echo ""
echo "=========================================="
echo "Completed: $SUBSET"
echo "=========================================="

if [ -f "data/${SUBSET}.pkl" ]; then
    SIZE=$(du -h "data/${SUBSET}.pkl" | cut -f1)
    echo "✓ Output file created: data/${SUBSET}.pkl ($SIZE)"
else
    echo "✗ ERROR: Output file not created"
    exit 1
fi
