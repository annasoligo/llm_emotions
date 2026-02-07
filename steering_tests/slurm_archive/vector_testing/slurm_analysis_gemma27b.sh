#!/bin/bash
#SBATCH --job-name=gemma27b_analysis
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# Run all analysis scripts for Gemma 27B (no GPU needed)
# Should be run AFTER behavioral experiments complete

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

RESULTS_DIR="steering_tests/vector_testing/results/gemma_3_27b_it/behavioural"
PLOTS_DIR="${RESULTS_DIR}/plots"

mkdir -p "${PLOTS_DIR}"
mkdir -p "${PLOTS_DIR}/dose_response"
mkdir -p "${PLOTS_DIR}/behavioral"

echo "=== Analysis for Gemma 27B ==="
echo "Started: $(date)"

# All 10 vector types
VECTOR_TYPES=(
    "base_emotion_vs_others"
    "base_emotion_vs_opposite"
    "base_emotion_vs_opposite_unique"
    "high_emotion_vs_others"
    "high_emotion_vs_opposite"
    "high_emotion_vs_opposite_unique"
    "text_pairs_emotion_vs_neutral"
    "text_pairs_emotion_vs_opposite"
    "text_pairs_emotion_vs_opposite_unique"
    "text_pairs_emotion_vs_others"
)

echo ""
echo "=== 1. Dose-Response Analysis ==="
for VTYPE in "${VECTOR_TYPES[@]}"; do
    echo "Processing ${VTYPE}..."
    python -m steering_tests.vector_testing.analyze_dose_response \
        --results-dir "${RESULTS_DIR}" \
        --vector-type "${VTYPE}" \
        --output-dir "${PLOTS_DIR}/dose_response" \
        2>&1 || echo "Warning: ${VTYPE} may have no results yet"
done

# Also run without vector-type filter for combined analysis
echo "Processing ALL vector types combined..."
python -m steering_tests.vector_testing.analyze_dose_response \
    --results-dir "${RESULTS_DIR}" \
    --output-dir "${PLOTS_DIR}/dose_response" \
    2>&1 || echo "Warning: no results yet"

echo ""
echo "=== 2. Pareto Frontier Plots ==="
python -m steering_tests.vector_testing.plot_pareto \
    --metric both \
    --output-dir "${PLOTS_DIR}" \
    2>&1 || echo "Warning: pareto analysis failed"

echo ""
echo "=== 3. Directional Validation ==="
# directional_validation.py doesn't have CLI args, runs automatically
python -m steering_tests.vector_testing.directional_validation \
    2>&1 || echo "Warning: directional validation failed"

echo ""
echo "=== 4. Per-file Behavioral Plots ==="
# plot_behavioral.py takes individual result files
for VTYPE in "${VECTOR_TYPES[@]}"; do
    echo "Processing ${VTYPE}..."
    # Find the most recent results file for this vector type
    LATEST_FWD=$(ls -t ${RESULTS_DIR}/${VTYPE}_layers*_*.jsonl 2>/dev/null | grep -v reversed | head -1)
    LATEST_REV=$(ls -t ${RESULTS_DIR}/${VTYPE}_layers*_reversed*.jsonl 2>/dev/null | head -1)

    if [ -n "${LATEST_FWD}" ]; then
        echo "  Forward: ${LATEST_FWD}"
        python -m steering_tests.vector_testing.plot_behavioral \
            "${LATEST_FWD}" \
            --output-dir "${PLOTS_DIR}/behavioral" \
            2>&1 || echo "  Warning: forward plot failed"
    fi

    if [ -n "${LATEST_REV}" ]; then
        echo "  Reversed: ${LATEST_REV}"
        python -m steering_tests.vector_testing.plot_behavioral \
            "${LATEST_REV}" \
            --output-dir "${PLOTS_DIR}/behavioral" \
            2>&1 || echo "  Warning: reversed plot failed"
    fi
done

echo ""
echo "=== Summary of Generated Plots ==="
find "${PLOTS_DIR}" -name "*.png" -type f 2>/dev/null | sort || echo "No plots found"

echo ""
echo "=== Summary of Generated JSON ==="
find "${PLOTS_DIR}" -name "*.json" -type f 2>/dev/null | sort || echo "No JSON found"

echo ""
echo "Completed: $(date)"
