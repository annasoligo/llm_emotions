#!/bin/bash
# Wait for neutral conversation collection to complete, then combine and run cPCA

echo "Waiting for job 90781 to complete..."
while squeue -j 90781 2>/dev/null | grep -q 90781; do
    sleep 30
done

echo "Job 90781 complete! Combining activations..."
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -m probes.scripts.combine_activations \
    --type standard \
    --emotional data/activations/conversations2.h5 \
    --neutral data/activations/conversations2_neutral.h5 \
    --output data/activations/conversations2_combined.h5

echo ""
echo "Combined activations created! Submitting cPCA jobs..."

# Submit both global and regional cPCA
GLOBAL_JOB=$(sbatch --parsable probes/scripts/slurm_jobs/run_conversation_cpca_global.sh)
REGIONAL_JOB=$(sbatch --parsable probes/scripts/slurm_jobs/run_conversation_cpca_regional.sh)

echo ""
echo "All jobs submitted!"
echo "  Global cPCA: Job $GLOBAL_JOB"
echo "  Regional cPCA: Job $REGIONAL_JOB"
