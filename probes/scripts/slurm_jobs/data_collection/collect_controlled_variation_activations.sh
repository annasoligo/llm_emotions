#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=cv_acts
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Collect activations from controlled variation data
# Creates two HDF5 files: user_isolation.h5 and assistant_isolation.h5

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python probes/scripts/data_collection/collect_controlled_variation_activations.py \
    --input outputs/data/controlled_variation/controlled_variation_all.jsonl \
    --output-dir outputs/activations/controlled_variation/ \
    --model google/gemma-3-27b-it

echo ""
echo "Activation extraction complete!"
