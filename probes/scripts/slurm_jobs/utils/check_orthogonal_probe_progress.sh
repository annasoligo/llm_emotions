#!/bin/bash
# Check progress of orthogonal probe training

RESULTS_DIR="/workspace-vast/annas/git/research-tools/probes/results/conversation_probes_orthogonal"

echo "========================================================================"
echo "ORTHOGONAL PROBE TRAINING PROGRESS"
echo "========================================================================"
echo ""
echo "Expected: 35 probes (5 layers × 7 representations)"
echo ""

# Count completed probes
COMPLETED=$(find "$RESULTS_DIR" -name "probe_layer*_ortho*.pkl" 2>/dev/null | wc -l)
echo "Completed: $COMPLETED / 35"
echo ""

# Show breakdown by layer
for LAYER in 10 20 30 40 50; do
    COUNT=$(find "$RESULTS_DIR" -name "probe_layer${LAYER}_*_ortho*.pkl" 2>/dev/null | wc -l)
    echo "Layer $LAYER: $COUNT / 7 representations"
done

echo ""
echo "========================================================================"
echo "RECENT COMPLETIONS (last 10):"
echo "========================================================================"
find "$RESULTS_DIR" -name "probe_layer*_ortho*.pkl" -type f -printf '%T@ %p\n' 2>/dev/null | \
    sort -rn | head -10 | while read timestamp file; do
    filename=$(basename "$file")
    date=$(date -d "@$timestamp" "+%Y-%m-%d %H:%M:%S")
    echo "$date - $filename"
done

echo ""
echo "========================================================================"
