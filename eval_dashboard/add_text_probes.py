"""
Add text_raw probe scores to existing preprocessed data.
"""
import pickle
import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from probes.scripts.probe_pipeline import TokenLevelExperiment
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from eval_dashboard.probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from eval_dashboard.data_preprocessing import (
    extract_activations,
    apply_probes_to_activations
)
from eval_dashboard.sentence_aggregator import aggregate_scores_to_sentences

# Load existing data
print("Loading existing preprocessed data...")
with open('eval_dashboard/data/preprocessed_conversations.pkl', 'rb') as f:
    data = pickle.load(f)

conversations = data['conversations']
print(f"Loaded {len(conversations)} conversations")
print(f"Existing probe types: {list(conversations[0]['probe_scores'].keys())}")

# Load model and tokenizer
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG['model_name'],
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# Compute text_raw baseline
print("\nComputing text_raw baseline...")
probe_config = PROBE_CONFIGS['text_raw']

baseline_loader = WildChatBaselineLoader(
    model=model,
    tokenizer=tokenizer,
    dataset_path=BASELINE_CONFIG['dataset_path'],
    num_samples=BASELINE_CONFIG['num_samples'],
    probe_dir=probe_config.get('probe_dir'),
    probe_pattern=probe_config.get('probe_pattern'),
    cpca_path=probe_config.get('cpca_path'),
    n_components=probe_config.get('n_components', 0),
    seed=probe_config.get('seed', 0),
    emotions=EMOTIONS
)

exp_temp = TokenLevelExperiment(
    model=model,
    tokenizer=tokenizer,
    probe_type=probe_config['type'],
    probe_dir=probe_config.get('probe_dir'),
    cpca_path=probe_config.get('cpca_path'),
    probe_pattern=probe_config.get('probe_pattern'),
    n_components=probe_config.get('n_components', 0),
    seed=probe_config.get('seed', 0),
    emotions=EMOTIONS
)

baseline_stats = baseline_loader.compute_probe_score_baselines(
    probe_inference=exp_temp.inference,
    layers=MODEL_CONFIG['layers'],
    probe_type=probe_config['type'],
    aggregation="mean",
    return_std=True,
    n_components=probe_config.get('n_components', 0),
    seed=probe_config.get('seed', 0),
    emotions=EMOTIONS
)

probe_baseline = {
    'mean': baseline_stats['mean'][-1],
    'std': baseline_stats['std'][-1]
}

# Initialize probe experiment
print("Initializing text_raw probe...")
probe_experiment = TokenLevelExperiment(
    model=None,
    tokenizer=tokenizer,
    probe_type=probe_config['type'],
    probe_dir=probe_config.get('probe_dir'),
    cpca_path=probe_config.get('cpca_path'),
    probe_pattern=probe_config.get('probe_pattern'),
    n_components=probe_config.get('n_components', 0),
    seed=probe_config.get('seed', 0),
    emotions=EMOTIONS
)

# Process each conversation
print(f"\nAdding text_raw scores to {len(conversations)} conversations...")
for i, conv in enumerate(conversations):
    print(f"  Processing {i+1}/{len(conversations)}...", end='\r')

    conversation = conv['conversation']
    sentences = [dict(s) for s in conv['sentences']]

    # Re-extract activations (we need these for text probes)
    activations_by_token, token_strings = extract_activations(
        model=model,
        tokenizer=tokenizer,
        conversation=conversation,
        layers=MODEL_CONFIG['layers']
    )

    # Apply text_raw probe
    token_scores = apply_probes_to_activations(
        activations_by_token=activations_by_token,
        probe_experiment=probe_experiment,
        layers=MODEL_CONFIG['layers']
    )

    # Normalize with baseline
    probe_mean = probe_baseline['mean']
    probe_std = probe_baseline['std']

    for token_pos in token_scores:
        score = token_scores[token_pos]
        token_scores[token_pos] = (score - probe_mean) / (probe_std + 1e-8)

    # Aggregate to sentences
    from eval_dashboard.sentence_aggregator import SentenceInfo
    sentence_infos = [SentenceInfo(**s) for s in sentences]

    sentence_scores = aggregate_scores_to_sentences(
        sentences=sentence_infos,
        token_scores=token_scores,
        aggregation='mean'
    )

    # Add to conversation
    conv['probe_scores']['text_raw'] = sentence_scores

print(f"\n✓ Added text_raw scores to all conversations")

# Save updated data
print("\nSaving updated data...")
with open('eval_dashboard/data/preprocessed_conversations.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✓ Done!")
