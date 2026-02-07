#!/bin/bash
#SBATCH --job-name=qwen32b_analysis
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# Run all analysis scripts for Qwen 32B (no GPU needed)

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

RESULTS_DIR="steering_tests/vector_testing/results/qwen2.5_32b_instruct/behavioural"
PLOTS_DIR="${RESULTS_DIR}/plots"

mkdir -p "${PLOTS_DIR}"
mkdir -p "${PLOTS_DIR}/dose_response"
mkdir -p "${PLOTS_DIR}/behavioral"

echo "=== Analysis for Qwen 32B ==="
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

# Combined analysis
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
python -m steering_tests.vector_testing.directional_validation \
    2>&1 || echo "Warning: directional validation failed"

echo ""
echo "=== 4. Vector Comparison Plot ==="
# Update plot script to handle Qwen 32B
python -c "
from steering_tests.vector_testing.plot_vector_comparison import *
model_dir = RESULTS_DIR / 'qwen2.5_32b_instruct'
print('Loading dose-response analysis for Qwen 32B...')
all_data = load_dose_response_analysis(model_dir)
print(f'Found {len(all_data)} vector types')
all_metrics = {}
for vtype, data in all_data.items():
    metrics = compute_metrics(data)
    if metrics:
        all_metrics[vtype] = metrics
output_path = model_dir / 'behavioural' / 'plots' / 'vector_type_comparison.png'
plot_comparison(all_metrics, output_path)
print_ranking(all_metrics)
" 2>&1 || echo "Warning: comparison plot failed"

echo ""
echo "=== Summary of Generated Plots ==="
find "${PLOTS_DIR}" -name "*.png" -type f 2>/dev/null | sort || echo "No plots found"

echo ""
echo "Completed: $(date)"
