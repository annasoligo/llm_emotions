#!/bin/bash
# Check progress of WildChat baseline extraction job

echo "================================"
echo "WildChat Baseline Job Status"
echo "================================"
echo ""

squeue -u annas | grep wildchat

echo ""
echo "================================"
echo "Recent Log Output"
echo "================================"

for log in /workspace-vast/annas/git/research-tools/probes/logs/wildchat_baseline_*.log; do
    if [ -f "$log" ]; then
        echo ""
        echo "=== $(basename $log) ==="
        tail -30 "$log"
    fi
done

echo ""
echo "================================"
echo "Output Files Created"
echo "================================"

output_dir="/workspace-vast/annas/git/research-tools/data/baselines/wildchat/google_gemma_3_27b_it"
if [ -d "$output_dir" ]; then
    echo ""
    echo "Activation files:"
    ls -lh "$output_dir"/*.h5 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

    echo ""
    echo "Summary stats:"
    ls -lh "$output_dir"/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
else
    echo "  No output files yet"
fi
