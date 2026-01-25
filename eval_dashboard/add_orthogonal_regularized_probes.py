"""
Add orthogonal_regularized probe scores to existing preprocessed data.
"""
import pickle
import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from probes.scripts.token_level_helpers import TokenLevelExperiment
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from eval_dashboard.probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from eval_dashboard.data_preprocessing import (
    extract_activations_for_conversation,
    apply_probes_to_activations
)
from eval_dashboard.sentence_aggregator import aggregate_scores_to_sentences, SentenceInfo

# Load existing data
print("="*80)
print("ADDING ORTHOGONAL REGULARIZED PROBES TO PREPROCESSED DATA")
print("="*80)

print("\nLoading existing preprocessed data...")
with open('data/preprocessed_conversations.pkl', 'rb') as f:
    data = pickle.load(f)

conversations = data['conversations']
print(f"✓ Loaded {len(conversations)} conversations")
print(f"  Existing probe types: {list(conversations[0]['probe_scores'].keys())}")

# Check if orthogonal_regularized_lambda100 already exists
if 'orthogonal_regularized_lambda100' in conversations[0]['probe_scores']:
    print("\n⚠ orthogonal_regularized_lambda100 scores already exist!")
    response = input("Do you want to overwrite? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

# Load model and tokenizer
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG['model_name'],
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print(f"✓ Model loaded on {model.device}")

# Compute orthogonal_regularized baseline
print("\nComputing orthogonal_regularized baseline...")
probe_key = 'orthogonal_regularized_lambda100'
probe_config = PROBE_CONFIGS[probe_key]

print(f"  Probe config:")
print(f"    Type: {probe_config['type']}")
print(f"    Dir: {probe_config['probe_dir']}")
print(f"    Pattern: {probe_config['probe_pattern']}")
print(f"    Lambda: {probe_config['lambda_ortho']}")

# Create temporary experiment to get baseline stats
exp_temp = TokenLevelExperiment(
    model=None,
    tokenizer=tokenizer,
    probe_type=probe_config['type'],
    probe_dir=probe_config.get('probe_dir'),
    cpca_path=probe_config.get('cpca_path'),
    probe_pattern=probe_config.get('probe_pattern'),
    lambda_ortho=probe_config.get('lambda_ortho', 100.0),
    emotions=EMOTIONS
)

print(f"\n  Computing baseline statistics...")
baseline_loader = WildChatBaselineLoader(
    aggregation_type=BASELINE_CONFIG['aggregation_type'],
    baseline_dir=BASELINE_CONFIG['baseline_dir']
)

baseline_stats = baseline_loader.compute_probe_score_baselines(
    probe_inference=exp_temp.inference,
    layers=MODEL_CONFIG['layers'],
    probe_type=probe_config['type'],
    aggregation="mean",
    return_std=True,
    emotions=EMOTIONS
)

probe_baseline = {
    'mean': baseline_stats['mean'][-1],  # Averaged across layers
    'std': baseline_stats['std'][-1]
}

print(f"  ✓ Baseline computed")
print(f"    Mean: {probe_baseline['mean']}")
print(f"    Std: {probe_baseline['std']}")

# Initialize probe experiment
print("\nInitializing orthogonal_regularized probe...")
probe_experiment = TokenLevelExperiment(
    model=None,
    tokenizer=tokenizer,
    probe_type=probe_config['type'],
    probe_dir=probe_config.get('probe_dir'),
    cpca_path=probe_config.get('cpca_path'),
    probe_pattern=probe_config.get('probe_pattern'),
    lambda_ortho=probe_config.get('lambda_ortho', 100.0),
    emotions=EMOTIONS
)

print("✓ Probe experiment initialized")

# Process each conversation
print(f"\nAdding orthogonal_regularized scores to {len(conversations)} conversations...")
from tqdm import tqdm

for i, conv in enumerate(tqdm(conversations, desc="Processing")):
    conversation = conv['conversation']
    sentences = [SentenceInfo(**s) for s in conv['sentences']]

    # Re-extract activations (we need these for text probes)
    activations_by_token, token_strings = extract_activations_for_conversation(
        conversation=conversation,
        model=model,
        tokenizer=tokenizer,
        layers=MODEL_CONFIG['layers']
    )

    # Apply orthogonal_regularized probe
    token_scores = apply_probes_to_activations(
        activations_by_token=activations_by_token,
        probe_experiment=probe_experiment,
        layers=MODEL_CONFIG['layers']
    )

    # Normalize with baseline (z-score)
    probe_mean = probe_baseline['mean']
    probe_std = probe_baseline['std']

    for token_pos in token_scores:
        score = token_scores[token_pos]
        token_scores[token_pos] = (score - probe_mean) / (probe_std + 1e-8)

    # Aggregate to sentences
    sentence_scores = aggregate_scores_to_sentences(
        sentences=sentences,
        token_scores=token_scores,
        aggregation='mean'
    )

    # Add to conversation
    conv['probe_scores'][probe_key] = sentence_scores

print(f"\n✓ Added {probe_key} scores to all conversations")

# Update probe configs in data
data['probe_configs'][probe_key] = probe_config
data['probe_baselines'][probe_key] = probe_baseline

# Save updated data
print("\nSaving updated data...")
with open('data/preprocessed_conversations.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✓ Data saved")

# Verify
print("\nVerifying...")
with open('data/preprocessed_conversations.pkl', 'rb') as f:
    data_verify = pickle.load(f)

print(f"  Probe types in data: {list(data_verify['probe_configs'].keys())}")
print(f"  Sample conversation has scores for: {list(data_verify['conversations'][0]['probe_scores'].keys())}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nOrthogonal regularized probes (λ=100, k=20) have been added to the dashboard data.")
print(f"You can now view them in the dashboard by selecting '{probe_config['display_name']}'.")
