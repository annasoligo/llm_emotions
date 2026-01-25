#!/bin/bash
# Submit all evaluation jobs

cd /workspace-vast/annas/git/research-tools/elicitation

echo "Submitting evaluation jobs..."

# Submit vanilla first (can run immediately)
JOB_VANILLA=$(sbatch --parsable slurm_eval_vanilla.sh)
echo "Vanilla job submitted: $JOB_VANILLA"

# Check if finetuned models exist
if [ -d "/workspace-vast/annas/models/gemma3-27b-lowfrust-lr1e4" ]; then
    if [ -f "/workspace-vast/annas/models/gemma3-27b-lowfrust-lr1e4/adapter_config.json" ]; then
        JOB_LR1E4=$(sbatch --parsable slurm_eval_lr1e4.sh)
        echo "LR1e4 job submitted: $JOB_LR1E4"
    else
        echo "LR1e4 model not ready (no adapter_config.json)"
    fi
else
    echo "LR1e4 model directory not found"
fi

if [ -d "/workspace-vast/annas/models/gemma3-27b-lowfrust-lr5e5" ]; then
    if [ -f "/workspace-vast/annas/models/gemma3-27b-lowfrust-lr5e5/adapter_config.json" ]; then
        JOB_LR5E5=$(sbatch --parsable slurm_eval_lr5e5.sh)
        echo "LR5e5 job submitted: $JOB_LR5E5"
    else
        echo "LR5e5 model not ready (no adapter_config.json)"
    fi
else
    echo "LR5e5 model directory not found"
fi

echo ""
echo "Submitted jobs:"
squeue -u annas --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
