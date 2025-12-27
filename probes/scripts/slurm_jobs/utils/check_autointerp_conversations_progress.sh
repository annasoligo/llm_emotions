#!/bin/bash
# Check progress of conversation autointerp jobs

echo "=== SLURM Job Status ==="
squeue -u annas | grep -E "autointe|JOBID"

echo ""
echo "=== Autointerp Output Files ==="
ls -lh /workspace-vast/annas/git/research-tools/probes/results/autointerp/conversations_*.json 2>/dev/null || echo "No output files yet"

echo ""
echo "=== Recent Log Output (Global) ==="
latest_global=$(ls -t /workspace-vast/annas/logs/*_autointerp_conv_global.out 2>/dev/null | head -1)
if [ -n "$latest_global" ]; then
    echo "File: $latest_global"
    tail -20 "$latest_global"
else
    echo "No global log file yet"
fi

echo ""
echo "=== Recent Log Output (Regional User) ==="
latest_user=$(ls -t /workspace-vast/annas/logs/*_autointerp_conv_user.out 2>/dev/null | head -1)
if [ -n "$latest_user" ]; then
    echo "File: $latest_user"
    tail -20 "$latest_user"
else
    echo "No user log file yet"
fi

echo ""
echo "=== Checkpoint Files ==="
ls -lh /workspace-vast/annas/git/research-tools/probes/results/autointerp/conversations_*_checkpoint.json 2>/dev/null || echo "No checkpoint files yet"

echo ""
echo "=== To check specific job logs ==="
echo "tail -f /workspace-vast/annas/logs/92597_autointerp_conv_global.out"
echo "tail -f /workspace-vast/annas/logs/92598_autointerp_conv_user.out"
echo "tail -f /workspace-vast/annas/logs/92599_autointerp_conv_asst.out"
echo "tail -f /workspace-vast/annas/logs/92600_autointerp_conv_special1.out"
echo "tail -f /workspace-vast/annas/logs/92601_autointerp_conv_special2.out"
