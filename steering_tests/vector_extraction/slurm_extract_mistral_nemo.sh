#!/bin/bash
#SBATCH --job-name=extract_mistral
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/extract_mistral_nemo_%j.out
#SBATCH --error=/workspace-vast/annas/logs/extract_mistral_nemo_%j.out

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

VECTORS_DIR="steering_tests/vectors/mistral_nemo"
mkdir -p "$VECTORS_DIR"

echo "====================================================="
echo "Mistral Nemo Vector Extraction"
echo "====================================================="

# 1. Base - emotion vs others
echo "[1/6] Base: emotion_vs_others"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/mistral_nemo_emotion_prompts \
  --output "$VECTORS_DIR/base_emotion_vs_others" \
  --method emotion_vs_others \
  --representation last_token

# 2. Base - emotion vs opposite
echo "[2/6] Base: emotion_vs_opposite"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/mistral_nemo_emotion_prompts \
  --output "$VECTORS_DIR/base_emotion_vs_opposite" \
  --method emotion_vs_opposite \
  --representation last_token

# 3. High - emotion vs others
echo "[3/6] High: emotion_vs_others"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/mistral_nemo_high_emotion_prompts \
  --output "$VECTORS_DIR/high_emotion_vs_others" \
  --method emotion_vs_others \
  --representation last_token

# 4. High - emotion vs opposite
echo "[4/6] High: emotion_vs_opposite"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/mistral_nemo_high_emotion_prompts \
  --output "$VECTORS_DIR/high_emotion_vs_opposite" \
  --method emotion_vs_opposite \
  --representation last_token

# 5. Text pairs - emotion vs opposite
echo "[5/6] Text pairs: emotion_vs_opposite"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/mistral_nemo_text_pairs \
  --output "$VECTORS_DIR/text_pairs_emotion_vs_opposite" \
  --method emotion_vs_opposite

# 6. Text pairs - emotion vs neutral
echo "[6/6] Text pairs: emotion_vs_neutral"
python steering_tests/vector_extraction/extract_directions.py \
  --activations steering_tests/activations/mistral_nemo_text_pairs \
  --output "$VECTORS_DIR/text_pairs_emotion_vs_neutral" \
  --method emotion_vs_neutral

echo "====================================================="
echo "DONE - Vectors in: $VECTORS_DIR"
ls -la "$VECTORS_DIR"
