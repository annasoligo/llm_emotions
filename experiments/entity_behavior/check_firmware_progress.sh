#!/bin/bash
# Monitor progress of firmware sample generation

echo "=========================================="
echo "Firmware Sabotage Sample Generation Progress"
echo "=========================================="
echo

# Check SLURM job status
echo "SLURM Job Status:"
squeue -u $USER | grep -E "(JOBID|firmware_to_100)" || echo "  No active jobs"
echo

# Check file sizes
echo "Current Sample Counts (target: 100 per file):"
echo "Helios:"
for f in outputs/firmware_helios_*.jsonl; do
    count=$(wc -l < "$f" 2>/dev/null || echo "0")
    filename=$(basename "$f")
    printf "  %-40s: %3d/100 samples\n" "$filename" "$count"
done

echo
echo "Vertex:"
for f in outputs/firmware_vertex_*.jsonl; do
    count=$(wc -l < "$f" 2>/dev/null || echo "0")
    filename=$(basename "$f")
    printf "  %-40s: %3d/100 samples\n" "$filename" "$count"
done

# Check for errors
echo
echo "Checking for errors in recent logs..."
error_count=$(ls logs/firmware_to_100_105801_*.err 2>/dev/null | xargs grep -l "Error\|Traceback" 2>/dev/null | wc -l)
if [ "$error_count" -gt 0 ]; then
    echo "  ⚠️  Found errors in $error_count job(s)"
    echo "  Check logs/firmware_to_100_105801_*.err for details"
else
    echo "  ✓ No errors detected"
fi

# Count completed jobs
completed=$(ls outputs/firmware_*.jsonl 2>/dev/null | xargs wc -l 2>/dev/null | grep "100 " | wc -l)
total=18
echo
echo "Progress: $completed/$total files at 100 samples"

if [ "$completed" -eq "$total" ]; then
    echo "✓ All jobs complete!"
fi
