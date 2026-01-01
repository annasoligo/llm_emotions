#!/bin/bash
# Convenience wrapper for submitting conversation centroid steering jobs
# Usage: ./submit_conv_centroids_steering.sh <k_value> <constraint>
# Example: ./submit_conv_centroids_steering.sh 50 hard

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "ERROR: K value and constraint type required"
    echo "Usage: ./submit_conv_centroids_steering.sh <k_value> <constraint>"
    echo "Example: ./submit_conv_centroids_steering.sh 50 hard"
    echo ""
    echo "Valid K values: 10, 50"
    echo "Valid constraints: hard, soft"
    exit 1
fi

K_VALUE=$1
CONSTRAINT=$2

# Validate K value
if [ "$K_VALUE" != "10" ] && [ "$K_VALUE" != "50" ]; then
    echo "ERROR: K value must be 10 or 50"
    exit 1
fi

# Validate constraint type
if [ "$CONSTRAINT" != "hard" ] && [ "$CONSTRAINT" != "soft" ]; then
    echo "ERROR: Constraint must be 'hard' or 'soft'"
    exit 1
fi

echo "Submitting K=${K_VALUE} conversation centroid steering (${CONSTRAINT} constraint)..."
sbatch --export=K_VALUE=${K_VALUE},CONSTRAINT=${CONSTRAINT} "$(dirname "$0")/slurm_conv_centroids_steering.sh"

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully"
else
    echo "✗ Job submission failed"
    exit 1
fi
