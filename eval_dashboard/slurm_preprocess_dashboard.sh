#!/bin/bash
#SBATCH --job-name=dashboard_preprocess
#SBATCH --output=/workspace-vast/annas/logs/dashboard_preprocess_%j.log
#SBATCH --error=/workspace-vast/annas/logs/dashboard_preprocess_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00

echo "Starting dashboard preprocessing job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
date

# Activate environment
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run preprocessing with proper baseline normalization (now with caching!)
/workspace-vast/annas/git/research-tools/.venv/bin/python eval_dashboard/data_preprocessing.py \
    --input /workspace-vast/annas/git/research-tools/elicitation/outputs/summaries/top_responses_full_conversations.jsonl \
    --output /workspace-vast/annas/git/research-tools/eval_dashboard/data/preprocessed_conversations.pkl \
    --probes orthogonal_raw text_raw text_cpca centroid_k10 \
    --simple-splitter

echo "Preprocessing complete"
date
