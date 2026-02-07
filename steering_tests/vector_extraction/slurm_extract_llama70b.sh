#!/bin/bash
#SBATCH --job-name=extract_llama70b
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/extract_llama70b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/extract_llama70b_%j.out

# Vector extraction for Llama 3.3 70B
# Run this AFTER activation collection jobs complete

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

VECTORS_DIR="steering_tests/vectors/llama70b"
mkdir -p "$VECTORS_DIR"

echo "====================================================="
echo "Llama 3.3 70B Vector Extraction"
echo "====================================================="

# 1. Base emotion prompts - emotion vs others
echo ""
echo "[1/6] Base emotion prompts: emotion_vs_others (last_token)"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/llama70b_emotion_prompts \
  --output "$VECTORS_DIR/base_emotion_vs_others" \
  --method emotion_vs_others \
  --representation last_token

# 2. Base emotion prompts - emotion vs opposite
echo ""
echo "[2/6] Base emotion prompts: emotion_vs_opposite (last_token)"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/llama70b_emotion_prompts \
  --output "$VECTORS_DIR/base_emotion_vs_opposite" \
  --method emotion_vs_opposite \
  --representation last_token

# 3. High emotion prompts - emotion vs others
echo ""
echo "[3/6] High emotion prompts: emotion_vs_others (last_token)"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/llama70b_high_emotion_prompts \
  --output "$VECTORS_DIR/high_emotion_vs_others" \
  --method emotion_vs_others \
  --representation last_token

# 4. High emotion prompts - emotion vs opposite
echo ""
echo "[4/6] High emotion prompts: emotion_vs_opposite (last_token)"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/llama70b_high_emotion_prompts \
  --output "$VECTORS_DIR/high_emotion_vs_opposite" \
  --method emotion_vs_opposite \
  --representation last_token

# 5. Text pairs - emotion vs opposite
echo ""
echo "[5/6] Text pairs: emotion_vs_opposite"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/llama70b_text_pairs \
  --output "$VECTORS_DIR/text_pairs_emotion_vs_opposite" \
  --method emotion_vs_opposite

# 6. Text pairs - emotion vs neutral
echo ""
echo "[6/6] Text pairs: emotion_vs_neutral"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/llama70b_text_pairs \
  --output "$VECTORS_DIR/text_pairs_emotion_vs_neutral" \
  --method emotion_vs_neutral

echo ""
echo "====================================================="
echo "EXTRACTION COMPLETE"
echo "Vectors saved to: $VECTORS_DIR"
echo "====================================================="
ls -la "$VECTORS_DIR"
