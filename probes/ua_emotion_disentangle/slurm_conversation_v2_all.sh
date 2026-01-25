#!/bin/bash
#SBATCH --job-name=conv_v2_all
#SBATCH --output=/workspace-vast/annas/logs/conversation_v2_all_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_v2_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=general

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

echo "========================================"
echo "CONVERSATION ANALYSIS V2 - ALL 6 CASES"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

# Layer 20
echo "=== Layer 20, Orthogonalized ==="
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v2.py \
    --layer 20 --method opposite --orthogonalize

echo ""
echo "=== Layer 20, Raw ==="
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v2.py \
    --layer 20 --method opposite

# Layer 30
echo ""
echo "=== Layer 30, Orthogonalized ==="
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v2.py \
    --layer 30 --method opposite --orthogonalize

echo ""
echo "=== Layer 30, Raw ==="
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v2.py \
    --layer 30 --method opposite

# Layer 40
echo ""
echo "=== Layer 40, Orthogonalized ==="
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v2.py \
    --layer 40 --method opposite --orthogonalize

echo ""
echo "=== Layer 40, Raw ==="
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v2.py \
    --layer 40 --method opposite

echo ""
echo "Completed: $(date)"
