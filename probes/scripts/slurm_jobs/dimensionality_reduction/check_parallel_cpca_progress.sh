#!/bin/bash

echo "=== Parallel Global cPCA Progress (Array Job 91263) ==="
echo ""

# Check how many tasks have completed
COMPLETED=0
RUNNING=0
FAILED=0

for i in {0..15}; do
    if squeue -j 91263_${i} &>/dev/null; then
        RUNNING=$((RUNNING + 1))
    else
        # Check if it completed successfully
        if [ -d "/workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_global/layers_$((i*4))_$((i*4+3))" ] || \
           [ -d "/workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_global/layers_60_61" ]; then
            COMPLETED=$((COMPLETED + 1))
        else
            # Check error log
            if grep -q "Error" /workspace-vast/annas/logs/91263_${i}.err 2>/dev/null; then
                FAILED=$((FAILED + 1))
            fi
        fi
    fi
done

echo "Status: $COMPLETED completed, $RUNNING running, $FAILED failed (out of 16 tasks)"
echo ""

# Show progress of each running task
echo "Running tasks:"
for i in {0..15}; do
    if squeue -j 91263_${i} &>/dev/null; then
        # Get last line of error output (progress bar)
        PROGRESS=$(tail -1 /workspace-vast/annas/logs/91263_${i}.err 2>/dev/null | grep -o "[0-9]\+%" | tail -1)
        if [ -n "$PROGRESS" ]; then
            echo "  Task $i (layers $((i*4))-$((i*4+3))): $PROGRESS"
        else
            echo "  Task $i (layers $((i*4))-$((i*4+3))): initializing..."
        fi
    fi
done
echo ""

# Check output directories
echo "Output directories created:"
ls -d /workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_global/layers_* 2>/dev/null | wc -l
echo "/16 layer ranges"
