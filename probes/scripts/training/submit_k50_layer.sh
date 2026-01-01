#!/bin/bash
# Convenience wrapper for submitting K=50 layer training jobs
# Usage: ./submit_k50_layer.sh <layer_number>
# Example: ./submit_k50_layer.sh 30

if [ -z "$1" ]; then
    echo "ERROR: Layer number required"
    echo "Usage: ./submit_k50_layer.sh <layer_number>"
    echo "Example: ./submit_k50_layer.sh 30"
    exit 1
fi

LAYER=$1

# Validate layer number is numeric
if ! [[ "$LAYER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Layer must be a number"
    exit 1
fi

echo "Submitting training job for K=50 Layer ${LAYER}..."
sbatch --export=LAYER=${LAYER} "$(dirname "$0")/slurm_k50_train_layer.sh"

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully"
else
    echo "✗ Job submission failed"
    exit 1
fi
