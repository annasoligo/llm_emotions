#!/bin/bash
# Launch comprehensive probe training sweep for conversation data
#
# Trains probes at layers: 10, 20, 30, 40, 50
# Representations:
#   - raw (mean activations)
#   - global_cpca with top 3, 5, 10 components
#   - regional_cpca with top 3, 5, 10 components per region (=12, 20, 40 total)
#
# All with high L1 sparsity (l1_lambda=0.1)

# Layers to train on
LAYERS=(10 20 30 40 50)

# L1 regularization for sparsity
L1_LAMBDA=0.1

cd /workspace-vast/annas/git/research-tools

echo "Launching conversation probe training sweep..."
echo "Layers: ${LAYERS[@]}"
echo "L1 lambda: $L1_LAMBDA"
echo ""

# Raw mean activations
echo "=== Raw mean activations ==="
for layer in "${LAYERS[@]}"; do
    echo "  Layer $layer: raw"
    sbatch probes/scripts/slurm_jobs/train_single_conversation_probe.sh \
        $layer raw 0 $L1_LAMBDA
done
echo ""

# Global cPCA - top 3, 5, 10
echo "=== Global cPCA ==="
for layer in "${LAYERS[@]}"; do
    for n_comp in 3 5 10; do
        echo "  Layer $layer: global_cpca top $n_comp"
        sbatch probes/scripts/slurm_jobs/train_single_conversation_probe.sh \
            $layer global_cpca $n_comp $L1_LAMBDA
    done
done
echo ""

# Regional cPCA - top 3, 5, 10 per region
echo "=== Regional cPCA ==="
for layer in "${LAYERS[@]}"; do
    for n_comp in 3 5 10; do
        echo "  Layer $layer: regional_cpca top $n_comp per region (= $((n_comp * 4)) total)"
        sbatch probes/scripts/slurm_jobs/train_single_conversation_probe.sh \
            $layer regional_cpca $n_comp $L1_LAMBDA
    done
done
echo ""

echo "Done! Launched $((5 + 5*3 + 5*3)) = 35 probe training jobs"
