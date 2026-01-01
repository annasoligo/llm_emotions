#!/bin/bash
# Monitor Dashboard Preprocessing SLURM Jobs

echo "=========================================="
echo "DASHBOARD PREPROCESSING STATUS"
echo "=========================================="
echo ""

# Check if job ID provided
if [ -z "$1" ]; then
    echo "Finding most recent preprocessing job..."
    LATEST_JOB=$(squeue -u $USER -n preprocess_dashboard -h -o "%A" | head -1)

    if [ -z "$LATEST_JOB" ]; then
        echo "No running preprocessing jobs found."
        echo ""
        echo "Checking recent logs..."
        ls -lt /workspace-vast/annas/logs/preprocess_dashboard_*.out 2>/dev/null | head -5
        exit 0
    fi

    JOB_ID=$LATEST_JOB
else
    JOB_ID=$1
fi

echo "Monitoring job: $JOB_ID"
echo ""

# Show job status
echo "Job Status:"
squeue -j $JOB_ID
echo ""

# Count completed
TOTAL=5
COMPLETED=$(ls /workspace-vast/annas/git/research-tools/eval_dashboard/data/*.pkl 2>/dev/null | wc -l)

echo "Progress: $COMPLETED / $TOTAL subsets completed"
echo ""

# Show which subsets are done
echo "Completed subsets:"
for FILE in high_emotion_6plus.pkl mid_emotion_3to5.pkl low_emotion_0to2.pkl low_emotion_no_shutdown.pkl low_emotion_with_shutdown.pkl; do
  if [ -f "/workspace-vast/annas/git/research-tools/eval_dashboard/data/$FILE" ]; then
    SIZE=$(du -h "/workspace-vast/annas/git/research-tools/eval_dashboard/data/$FILE" | cut -f1)
    echo "  ✓ $FILE ($SIZE)"
  else
    echo "  ⧗ $FILE (in progress or pending)"
  fi
done

echo ""
echo "Recent log output (array task 0):"
LOG_FILE="/workspace-vast/annas/logs/preprocess_dashboard_${JOB_ID}_0.out"
if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE"
else
    echo "  Log file not yet created"
fi

echo ""
echo "=========================================="
echo "To monitor all logs: tail -f /workspace-vast/annas/logs/preprocess_dashboard_${JOB_ID}_*.out"
echo "=========================================="
