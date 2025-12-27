#!/bin/bash
#SBATCH --job-name=eval_layer50
#SBATCH --output=/workspace-vast/annas/logs/eval_layer50_%j.log
#SBATCH --error=/workspace-vast/annas/logs/eval_layer50_%j.err
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

set -e

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Change to repo directory
cd /workspace-vast/annas/git/research-tools

# Configuration
MODEL="google/gemma-3-27b-it"
CONVERSATIONS="data/conversations2.jsonl"
LIMIT=200
START_TOKEN=20

echo "==========================================="
echo "EVALUATING LAYER 50 FOR ALL_CPCA"
echo "==========================================="
echo "Model: $MODEL"
echo "Conversations: $CONVERSATIONS (limit: $LIMIT)"
echo "Start token: $START_TOKEN"
echo ""

# Test layer 50 only
python probes/scripts/evaluation/eval_probes_on_conversations.py \
    --conversations $CONVERSATIONS \
    --model $MODEL \
    --probe-dir results/emotion_probes_high_alpha_cpca \
    --probe-pattern "probe_layer{layer}_all_cpca.pkl" \
    --layers 50 \
    --output results/conversation_eval/gemma3_all_cpca_layer50.json \
    --start-token $START_TOKEN \
    --limit $LIMIT \
    --probe-name "all_cpca_layer50"

echo ""
echo "==========================================="
echo "LAYER 50 EVALUATION COMPLETED"
echo "==========================================="
echo ""
echo "Results saved to:"
echo "  - results/conversation_eval/gemma3_all_cpca_layer50.json"
echo ""
echo "To merge with main results, run:"
echo "  python probes/scripts/merge_layer50_results.py"
