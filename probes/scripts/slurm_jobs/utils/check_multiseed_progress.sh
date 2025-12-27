#!/bin/bash
# Check progress of multi-seed probe training

echo "=========================================="
echo "MULTI-SEED PROBE TRAINING PROGRESS"
echo "=========================================="
echo ""

# Check SLURM queue
echo "=== SLURM Jobs ==="
squeue -u annas | grep probe_sweep | wc -l | xargs echo "Jobs in queue:"

# Count completed probes
PROBE_DIR="/workspace-vast/annas/git/research-tools/results/emotion_probes_multiseed"
if [ -d "$PROBE_DIR" ]; then
    echo ""
    echo "=== Completed Probes ==="
    total_probes=$(find $PROBE_DIR -name "probe_*.pkl" 2>/dev/null | wc -l)
    echo "Total probes saved: $total_probes / 400"

    echo ""
    echo "=== By Configuration ==="
    for nc in 0 5 10 20; do
        count=$(find $PROBE_DIR -name "probe_*_nc${nc}_seed*.pkl" 2>/dev/null | wc -l)
        label="nc=$nc"
        if [ "$nc" -eq 0 ]; then
            label="Raw"
        else
            label="${nc} PCs"
        fi
        echo "$label: $count / 100"
    done

    echo ""
    echo "=== By Layer ==="
    for layer in 5 10 15 20 25 30 35 40 45 50; do
        count=$(find $PROBE_DIR -name "probe_layer${layer}_*.pkl" 2>/dev/null | wc -l)
        echo "Layer $layer: $count / 40"
    done
else
    echo ""
    echo "Probe directory not found: $PROBE_DIR"
fi

echo ""
echo "=== Recent Logs ==="
ls -lt /workspace-vast/annas/logs/probe_sweep_*.log 2>/dev/null | head -5

echo ""
echo "=== Check a recent log ==="
latest_log=$(ls -t /workspace-vast/annas/logs/probe_sweep_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "Latest: $latest_log"
    tail -20 "$latest_log"
fi
