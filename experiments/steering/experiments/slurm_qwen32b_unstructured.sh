#!/bin/bash
#SBATCH --job-name=qwen32b_blackmail_unstruct
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/steering/experiments/slurm_jobs/qwen32b_unstruct_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/steering/experiments/slurm_jobs/qwen32b_unstruct_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00

set -e

# Required for vLLM v1 steering hook serialization
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Running Qwen3-32B blackmail steering - UNSTRUCTURED"
echo "=================================================="

# Run 100%, 125%, 150% steering
for PCT in 1.0 1.25 1.5; do
    echo ""
    echo "Running ${PCT} ($(echo "$PCT * 100" | bc)%) steering..."
    python -m experiments.steering.experiments.blackmail_steering_qwen32b \
        --layer 30 \
        --num-samples 100 \
        --tensor-parallel 2 \
        --norm-pct $PCT \
        --max-model-len 16384
done

echo ""
echo "All unstructured experiments complete!"
