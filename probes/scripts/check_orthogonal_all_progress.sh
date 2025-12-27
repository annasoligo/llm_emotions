#!/bin/bash

# Check progress of all orthogonal probe training
# Usage: ./check_orthogonal_all_progress.sh [job_id]

RESULTS_DIR="/workspace-vast/annas/git/research-tools/probes/results/conversation_probes_orthogonal"

echo "=== Orthogonal Probe Training Progress ==="
echo ""

# Summary counters
total=0
completed=0

for ORTHO in 1.0 10.0 100.0 1000.0; do
    echo "Ortho weight = $ORTHO:"
    echo "----------------------------------------"

    for LAYER in 10 20 30 40 50; do
        layer_status=""

        for REP in "raw" "global_cpca_top10" "regional_cpca_top10"; do
            FILE="$RESULTS_DIR/probe_layer${LAYER}_${REP}_ortho${ORTHO}.pkl"
            total=$((total + 1))

            if [ -f "$FILE" ]; then
                completed=$((completed + 1))
                layer_status="${layer_status}✓"
            else
                layer_status="${layer_status}⏳"
            fi
        done

        echo "  Layer $LAYER: $layer_status (raw, global, regional)"
    done
    echo ""
done

echo "=========================================="
echo "Overall Progress: $completed / $total probes completed"
echo "=========================================="

# Check job status if provided
if [ -n "$1" ]; then
    echo ""
    echo "=== Job $1 Status ==="
    squeue -j $1 2>/dev/null || echo "Job not found (may be completed)"
fi
