#!/bin/bash
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=emotion_onset_probes
#SBATCH --output=/workspace-vast/annas/logs/emotion_onset_probes_%j.out
#SBATCH --error=/workspace-vast/annas/logs/emotion_onset_probes_%j.err
#SBATCH --time=01:00:00

# Apply emotion probes to window activations
# This script does NOT need a GPU (no model inference)

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Starting emotion onset probe analysis..."
cd /workspace-vast/annas/git/research-tools/elicitation/scripts

# Run probe analysis (optional --probe-type argument: orthogonal, linear, or centroid)
PROBE_TYPE="${1:-orthogonal}"
echo "Using probe type: $PROBE_TYPE"

python run_emotion_onset_probes.py --probe-type "$PROBE_TYPE"
