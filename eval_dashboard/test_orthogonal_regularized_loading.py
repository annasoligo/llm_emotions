"""
Test loading and inference with orthogonal regularized probes.
"""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from pathlib import Path
import numpy as np
from probes.scripts.token_level_helpers import TokenLevelExperiment
from transformers import AutoTokenizer
from eval_dashboard.probe_configs import PROBE_CONFIGS, MODEL_CONFIG

print("="*80)
print("TESTING ORTHOGONAL REGULARIZED PROBE LOADING")
print("="*80)

# Get probe config
probe_key = 'orthogonal_regularized_lambda100'
probe_config = PROBE_CONFIGS[probe_key]

print(f"\nProbe config: {probe_key}")
print(f"  Type: {probe_config['type']}")
print(f"  Dir: {probe_config['probe_dir']}")
print(f"  Pattern: {probe_config['probe_pattern']}")
print(f"  Lambda: {probe_config['lambda_ortho']}")

# Load tokenizer (no model needed for this test)
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

# Initialize experiment
print("\nInitializing TokenLevelExperiment...")
experiment = TokenLevelExperiment(
    model=None,
    tokenizer=tokenizer,
    probe_type=probe_config['type'],
    probe_dir=probe_config['probe_dir'],
    probe_pattern=probe_config['probe_pattern'],
    lambda_ortho=probe_config['lambda_ortho'],
    emotions=['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
)

print("✓ Experiment initialized")

# Test loading a probe
print("\nTesting probe loading...")
layer = 30

probe_data = experiment.inference.load_orthogonal_regularized_probe(
    layer=layer,
    lambda_ortho=probe_config['lambda_ortho'],
    probe_pattern=probe_config['probe_pattern']
)

print(f"✓ Loaded probe for layer {layer}")
print(f"  Model: {type(probe_data['model'])}")
print(f"  Label names: {probe_data['label_names']}")

# Test inference with dummy activations
print("\nTesting inference with dummy activations...")
hidden_dim = 5376  # Gemma-3-27B hidden dim
dummy_activations = np.random.randn(1, hidden_dim).astype(np.float32)

scores = experiment.inference.predict_orthogonal_regularized(
    activations=dummy_activations,
    probe_model=probe_data['model'],
    label_names=probe_data['label_names'],
    emotions=experiment.emotions
)

print(f"✓ Inference successful")
print(f"  Output shape: {scores.shape}")
print(f"  Expected shape: (1, {len(experiment.emotions)})")
print(f"  Scores: {scores[0]}")

# Test loading probes for all layers
print("\nTesting probe loading for all layers...")
layers_to_test = [20, 25, 30, 35, 40]
for layer in layers_to_test:
    probe_data = experiment.inference.load_orthogonal_regularized_probe(
        layer=layer,
        lambda_ortho=probe_config['lambda_ortho'],
        probe_pattern=probe_config['probe_pattern']
    )
    print(f"  Layer {layer}: ✓")

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nOrthogonal regularized probes are ready for integration into the dashboard.")
