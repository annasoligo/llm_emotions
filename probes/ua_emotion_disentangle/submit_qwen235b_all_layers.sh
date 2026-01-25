#!/bin/bash
# Submit multiple jobs to collect all 94 layers of Qwen3-235B-A22B
# Split into chunks to keep file sizes manageable

cd /workspace-vast/annas/git/research-tools

# Layer chunks (94 layers total, ~19 layers each)
CHUNKS=(
    "0-18"
    "19-37"
    "38-56"
    "57-75"
    "76-93"
)

echo "============================================================"
echo "Submitting Qwen3-235B-A22B UA activation collection jobs"
echo "============================================================"
echo "Model: Qwen/Qwen3-235B-A22B (94 layers, 4096 hidden dim)"
echo "Chunks: ${#CHUNKS[@]}"
echo ""

for chunk in "${CHUNKS[@]}"; do
    echo "Submitting layers $chunk..."

    # Submit with layer range as environment variable
    job_id=$(LAYERS="$chunk" sbatch --parsable probes/ua_emotion_disentangle/slurm_collect_qwen_235b.sh)

    echo "  Job ID: $job_id (layers $chunk)"
done

echo ""
echo "============================================================"
echo "All jobs submitted! Monitor with: squeue -u \$USER"
echo "============================================================"
