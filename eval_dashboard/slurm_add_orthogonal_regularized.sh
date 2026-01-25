#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=add_ortho_reg
#SBATCH --output=/workspace-vast/annas/logs/add_ortho_reg_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/add_ortho_reg_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --array=0-5

# SLURM Array Job for Adding Orthogonal Regularized Probes to Dashboard Subsets
# Each array task processes one subset file
#
# Usage: sbatch slurm_add_orthogonal_regularized.sh

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

cd /workspace-vast/annas/git/research-tools/eval_dashboard

# Define subsets array
SUBSETS=(
    "high_emotion_6plus.pkl"
    "mid_emotion_3to5.pkl"
    "low_emotion_0to2.pkl"
    "low_emotion_no_shutdown.pkl"
    "low_emotion_with_shutdown.pkl"
    "baseline_v12_solvable.pkl"
)

SUBSET=${SUBSETS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Adding Orthogonal Regularized Probes"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Subset: $SUBSET"
echo "=========================================="

python add_orthogonal_one_subset.py $SUBSET

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed: $SUBSET"

    # Verify the file was updated
    python3 -c "
import pickle
with open('data/$SUBSET', 'rb') as f:
    data = pickle.load(f)
conv = data['conversations'][0]
if 'orthogonal_regularized_lambda100' in conv['probe_scores']:
    num_scores = len(conv['probe_scores']['orthogonal_regularized_lambda100'])
    print(f'✓ Verified: {num_scores} sentences have orthogonal_regularized scores')
else:
    print('✗ ERROR: orthogonal_regularized_lambda100 not found in probe_scores')
    exit(1)
"
else
    echo "✗ ERROR: Failed to process $SUBSET (exit code: $EXIT_CODE)"
    exit 1
fi
echo "=========================================="
