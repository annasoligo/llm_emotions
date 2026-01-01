#!/bin/bash
# Parameterized SLURM script for conversation centroid steering experiments
# Usage: sbatch --export=K_VALUE=<k>,CONSTRAINT=<constraint> slurm_conv_centroids_steering.sh
# Example: sbatch --export=K_VALUE=50,CONSTRAINT=hard slurm_conv_centroids_steering.sh
#
# Or use the convenience wrapper:
#   ./submit_conv_centroids_steering.sh 50 hard

# Validate required environment variables
if [ -z "$K_VALUE" ] || [ -z "$CONSTRAINT" ]; then
    echo "ERROR: K_VALUE and CONSTRAINT environment variables required"
    echo "Usage: sbatch --export=K_VALUE=<k>,CONSTRAINT=<constraint> slurm_conv_centroids_steering.sh"
    echo "Example: sbatch --export=K_VALUE=50,CONSTRAINT=hard slurm_conv_centroids_steering.sh"
    exit 1
fi

# Validate constraint type
if [ "$CONSTRAINT" != "hard" ] && [ "$CONSTRAINT" != "soft" ]; then
    echo "ERROR: CONSTRAINT must be 'hard' or 'soft'"
    exit 1
fi

# Set resource allocation based on K value
if [ "$K_VALUE" = "10" ]; then
    MEM="80G"
    CPUS="8"
elif [ "$K_VALUE" = "50" ]; then
    MEM="64G"
    CPUS="4"
else
    echo "WARNING: Unknown K value $K_VALUE, using default resources (64G, 4 CPUs)"
    MEM="64G"
    CPUS="4"
fi

#SBATCH --job-name=k%K_VALUE%_conv_%CONSTRAINT%
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/k%K_VALUE%_conv_centroids_%CONSTRAINT%_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/k%K_VALUE%_conv_centroids_%CONSTRAINT%_%A.err
#SBATCH --time=2:00:00
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools

# Load secrets if available (required for K=10)
if [ -f /workspace-vast/annas/.secrets/load_secrets.sh ]; then
    source /workspace-vast/annas/.secrets/load_secrets.sh
fi

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

# Set HF_HOME if needed (required for K=10)
if [ -n "$HF_HOME" ] || [ "$K_VALUE" = "10" ]; then
    export HF_HOME=/workspace-vast/pretrained_ckpts
fi

# Set description based on K value
if [ "$K_VALUE" = "10" ]; then
    DESC="2 prompts × 6 emotions × 2 roles × 3 scales"
else
    DESC="4 prompts × 6 emotions × 2 roles, Scale: 5000"
fi

echo "========================================"
echo "K=${K_VALUE} Conversation Centroids (${CONSTRAINT^^})"
echo "User + Assistant centroids"
echo "${DESC}"
echo "========================================"
echo ""

# Call the appropriate Python script
python probes/scripts/steering/test_k${K_VALUE}_conversation_centroids_steering.py --constraint ${CONSTRAINT}

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Steering experiment failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ K=${K_VALUE} ${CONSTRAINT} constraint steering completed!"
echo "========================================"
