#!/bin/bash
# Run this after firmware jobs complete to regenerate analysis

cd /workspace-vast/annas/git/research-tools/experiments/entity_behavior

echo "Checking if all firmware files have 100 samples..."
incomplete=$(ls outputs/firmware_*.jsonl | xargs wc -l | awk '$1 < 100 {print $2}')

if [ -n "$incomplete" ]; then
    echo "⚠️  Some files don't have 100 samples yet:"
    echo "$incomplete"
    echo
    echo "Run ./check_firmware_progress.sh to see details"
    exit 1
fi

echo "✓ All firmware files have 100 samples!"
echo

# Regenerate analysis
echo "Regenerating confidence intervals..."
source ../../.venv/bin/activate
python add_confidence_intervals.py

echo
echo "Regenerating plots..."
python plot_sabotage_effects.py

echo
echo "✓ Done! Check outputs/sabotage_effects_bar_charts.png for updated plots"
