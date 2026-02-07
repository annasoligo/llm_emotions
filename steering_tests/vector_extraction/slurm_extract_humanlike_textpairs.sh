#!/bin/bash
#SBATCH --job-name=extract_hl_pairs
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/extract_humanlike_pairs_%j.out
#SBATCH --error=/workspace-vast/annas/logs/extract_humanlike_pairs_%j.out

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

VECTORS_DIR="steering_tests/vectors/humanlike_mistral"

echo "====================================================="
echo "HumanLike Mistral Text Pairs Vector Extraction"
echo "====================================================="

# Text pairs - emotion vs opposite
echo "[1/2] Text pairs: emotion_vs_opposite"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/humanlike_mistral_text_pairs \
  --output "$VECTORS_DIR/text_pairs_emotion_vs_opposite" \
  --method emotion_vs_opposite

# Text pairs - emotion vs neutral
echo "[2/2] Text pairs: emotion_vs_neutral"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/humanlike_mistral_text_pairs \
  --output "$VECTORS_DIR/text_pairs_emotion_vs_neutral" \
  --method emotion_vs_neutral

echo "====================================================="
echo "DONE"
ls -la "$VECTORS_DIR"
