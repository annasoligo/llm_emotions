#!/bin/bash
#SBATCH --job-name=eval_probes
#SBATCH --output=/workspace-vast/annas/logs/eval_probes_%j.log
#SBATCH --error=/workspace-vast/annas/logs/eval_probes_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

# Generic probe evaluation job
#
# Usage:
#   # Single probe type
#   sbatch eval_probes.sh \
#       --probe-dir results/emotion_probes_raw \
#       --probe-pattern "probe_layer{layer}_all.pkl" \
#       --probe-name raw \
#       --output results/conversation_eval/gemma3_raw.json
#
#   # Multiple probe types (sweep mode)
#   sbatch eval_probes.sh --sweep \
#       --probe-configs "raw:results/emotion_probes_raw:probe_layer{layer}_all.pkl,top3:results/emotion_probes_top3:probe_layer{layer}_all_cpca_top3.pkl"

set -e

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Change to repo directory
cd /workspace-vast/annas/git/research-tools

# Default configuration
MODEL="google/gemma-3-27b-it"
CONVERSATIONS="data/conversations2.jsonl"
LIMIT=200
START_TOKEN=20
LAYERS="5 10 15 20 25 30 35 40 45 50"
SWEEP_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --conversations)
            CONVERSATIONS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --start-token)
            START_TOKEN="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --probe-dir)
            PROBE_DIR="$2"
            shift 2
            ;;
        --probe-pattern)
            PROBE_PATTERN="$2"
            shift 2
            ;;
        --probe-name)
            PROBE_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --sweep)
            SWEEP_MODE=true
            shift
            ;;
        --probe-configs)
            PROBE_CONFIGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "PROBE EVALUATION"
echo "=========================================="
echo "Model: $MODEL"
echo "Conversations: $CONVERSATIONS (limit: $LIMIT)"
echo "Start token: $START_TOKEN"
echo "Layers: $LAYERS"
echo ""

if [[ "$SWEEP_MODE" == true ]]; then
    # Sweep mode - evaluate multiple probe types
    echo "Running in SWEEP mode..."
    echo ""

    # Default configs if not provided
    if [[ -z "$PROBE_CONFIGS" ]]; then
        PROBE_CONFIGS="raw:results/emotion_probes_raw:probe_layer{layer}_all.pkl,all_cpca:results/emotion_probes_high_alpha_cpca:probe_layer{layer}_all_cpca.pkl,top3:results/emotion_probes_top3:probe_layer{layer}_all_cpca_top3.pkl,top5:results/emotion_probes_top5:probe_layer{layer}_all_cpca_top5.pkl,top10:results/emotion_probes_top10:probe_layer{layer}_all_cpca_top10.pkl,top20:results/emotion_probes_top20:probe_layer{layer}_all_cpca_top20.pkl"
    fi

    IFS=',' read -ra CONFIGS <<< "$PROBE_CONFIGS"
    TOTAL=${#CONFIGS[@]}

    for i in "${!CONFIGS[@]}"; do
        CONFIG="${CONFIGS[$i]}"
        IFS=':' read -r NAME DIR PATTERN <<< "$CONFIG"

        echo "$((i+1))/$TOTAL: Testing $NAME probes..."

        python probes/scripts/evaluation/eval_probes_on_conversations.py \
            --conversations "$CONVERSATIONS" \
            --model "$MODEL" \
            --probe-dir "$DIR" \
            --probe-pattern "$PATTERN" \
            --layers $LAYERS \
            --output "results/conversation_eval/gemma3_${NAME}.json" \
            --start-token "$START_TOKEN" \
            --limit "$LIMIT" \
            --probe-name "$NAME"

        echo ""
    done

    echo "=========================================="
    echo "ALL EVALUATIONS COMPLETED"
    echo "=========================================="
    echo ""
    echo "Results saved to results/conversation_eval/"
    echo ""
    echo "Generate visualizations:"
    echo "  python probes/scripts/visualization/visualize_all_conversation_eval.py"

else
    # Single evaluation mode
    if [[ -z "$PROBE_DIR" ]] || [[ -z "$PROBE_PATTERN" ]] || [[ -z "$OUTPUT" ]]; then
        echo "Error: For single evaluation, must provide --probe-dir, --probe-pattern, and --output"
        exit 1
    fi

    echo "Evaluating $PROBE_NAME probes..."
    echo "  Probe dir: $PROBE_DIR"
    echo "  Pattern: $PROBE_PATTERN"
    echo "  Output: $OUTPUT"
    echo ""

    python probes/scripts/evaluation/eval_probes_on_conversations.py \
        --conversations "$CONVERSATIONS" \
        --model "$MODEL" \
        --probe-dir "$PROBE_DIR" \
        --probe-pattern "$PROBE_PATTERN" \
        --layers $LAYERS \
        --output "$OUTPUT" \
        --start-token "$START_TOKEN" \
        --limit "$LIMIT" \
        --probe-name "${PROBE_NAME:-probe}"

    echo ""
    echo "=========================================="
    echo "EVALUATION COMPLETED"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT"
fi
