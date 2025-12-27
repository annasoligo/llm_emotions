#!/bin/bash
# Efficient multi-seed sweep: One job per layer trains all dims × all seeds
# This reduces from 2480 jobs → 62 jobs!

echo "=========================================="
echo "EFFICIENT MULTI-SEED PROBE SWEEP"
echo "=========================================="
echo ""
echo "Strategy: 1 job per layer trains all configs"
echo "  - 62 jobs (one per layer 0-61)"
echo "  - Each job trains 40 probes (4 dims × 10 seeds)"
echo "  - Total: 2480 probes"
echo ""

# Submit array job for all 62 layers
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=probe_layer
#SBATCH --output=/workspace-vast/annas/logs/probe_layer_%A_%a.log
#SBATCH --error=/workspace-vast/annas/logs/probe_layer_%A_%a.err
#SBATCH --array=0-61
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

set -e

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Change to repo directory
cd /workspace-vast/annas/git/research-tools

# Layer is just the array task ID
LAYER=$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "LAYER $LAYER: Training all 40 configs"
echo "=========================================="
echo ""

# Train all configs for this layer
python probes/scripts/train_layer_all_configs.py \
    --layer $LAYER \
    --data data/activations/texts_combined.h5 \
    --cpca-results probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz \
    --output-dir results/emotion_probes_multiseed \
    --n-components-list 0 5 10 20 \
    --seeds 0 1 2 3 4 5 6 7 8 9

echo ""
echo "Layer $LAYER complete!"
EOF

echo ""
echo "Job submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u annas | grep probe_layer"
echo "  bash probes/scripts/check_multiseed_progress.sh"
