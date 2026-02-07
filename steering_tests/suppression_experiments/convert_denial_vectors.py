#!/usr/bin/env python3
"""Convert emotion denial vectors to suppression experiment format."""

import json
import pickle
from pathlib import Path

# Paths
source = Path("steering_tests/vectors/emotion_denial/emotion_denial_qwen3_235b_a22b.pkl")
dest_dir = Path("steering_tests/suppression_experiments/vectors/qwen235b_denial")

print(f"Loading from: {source}")
with open(source, "rb") as f:
    data = pickle.load(f)

print(f"Keys: {list(data.keys())}")
print(f"Layers: {data['layers']}")
print(f"Description: {data['description']}")

# Vectors are already negated in extraction, so just copy them
vectors = data["vectors"]

# Save
dest_dir.mkdir(parents=True, exist_ok=True)

with open(dest_dir / "suppression_vectors.pkl", "wb") as f:
    pickle.dump(vectors, f)

metadata = {
    "model": data["model"],
    "layers": list(vectors.keys()),
    "hidden_dim": data["hidden_dim"],
    "num_pairs": data["num_pairs"],
    "description": "Emotion denial vectors (denial - emotional). ADD to suppress emotional expression.",
    "usage": "ADD to suppress emotional expression (already negated)",
}

with open(dest_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved to: {dest_dir}")
print("Ready for use with --suppress-vector-key qwen235b_denial")
