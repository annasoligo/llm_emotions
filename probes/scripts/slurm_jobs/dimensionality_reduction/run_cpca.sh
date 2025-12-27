#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=cpca
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Generic cPCA job - runs any cPCA config file
#
# Usage:
#   sbatch run_cpca.sh <config.yaml>
#   sbatch run_cpca.sh probes/experiments/configs/cpca_conversations_global.yaml
#
# Or pass config path as argument:
#   sbatch run_cpca.sh --config probes/experiments/configs/cpca_conversations_regional.yaml

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Parse arguments
CONFIG_PATH=""

# Check if config passed directly as first arg
if [[ $1 != --* ]] && [[ -n $1 ]]; then
    CONFIG_PATH="$1"
else
    # Parse --config flag
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_PATH="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: sbatch run_cpca.sh <config.yaml>"
                echo "   or: sbatch run_cpca.sh --config <config.yaml>"
                exit 1
                ;;
        esac
    done
fi

# Validate config path
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: No config file specified"
    echo "Usage: sbatch run_cpca.sh <config.yaml>"
    echo "   or: sbatch run_cpca.sh --config <config.yaml>"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "Running cPCA with config: $CONFIG_PATH"
echo ""

# Run cPCA
python -m probes.scripts.run_cpca "$CONFIG_PATH"

echo ""
echo "Done!"
