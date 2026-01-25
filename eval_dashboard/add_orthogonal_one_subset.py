"""
Add orthogonal_regularized scores to ONE subset file (memory efficient).
Usage: python3 add_orthogonal_one_subset.py <subset_file>
"""
import pickle
import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from eval_dashboard.probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from eval_dashboard.data_preprocessing import extract_activations_for_conversation, apply_probes_to_activations
from eval_dashboard.sentence_aggregator import aggregate_scores_to_sentences, SentenceInfo
from probes.scripts.token_level_helpers import TokenLevelExperiment
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from tqdm import tqdm

if len(sys.argv) != 2:
    print("Usage: python3 add_orthogonal_one_subset.py <subset_file>")
    sys.exit(1)

subset_file = sys.argv[1]
data_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data")
file_path = data_dir / subset_file

if not file_path.exists():
    print(f"Error: {file_path} not found")
    sys.exit(1)

print("="*80)
print(f"ADDING ORTHOGONAL REGULARIZED PROBES TO {subset_file}")
print("="*80)

# Get probe config
probe_key = 'orthogonal_regularized_lambda100'
probe_config = PROBE_CONFIGS[probe_key]

print(f"\nProbe config: {probe_key}")
print(f"  Type: {probe_config['type']}")
print(f"  Lambda: {probe_config['lambda_ortho']}")

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

# Compute baseline
print("\nComputing baseline statistics...")
exp_temp = TokenLevelExperiment(
    model=None,
    tokenizer=tokenizer,
    probe_type=probe_config['type'],
    probe_dir=probe_config.get('probe_dir'),
    probe_pattern=probe_config.get('probe_pattern'),
    lambda_ortho=probe_config.get('lambda_ortho', 100.0),
    emotions=EMOTIONS
)

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
    'mean': baseline_stats['mean'][-1],
    'std': baseline_stats['std'][-1]
}

print(f"✓ Baseline computed")

# Initialize probe experiment
print("\nInitializing probe experiment...")
probe_experiment = TokenLevelExperiment(
    model=None,
    tokenizer=tokenizer,
    probe_type=probe_config['type'],
    probe_dir=probe_config.get('probe_dir'),
    probe_pattern=probe_config.get('probe_pattern'),
    lambda_ortho=probe_config.get('lambda_ortho', 100.0),
    emotions=EMOTIONS
)
print("✓ Probe experiment initialized")

# Load subset data
print(f"\nLoading {subset_file}...")
with open(file_path, 'rb') as f:
    data = pickle.load(f)

conversations = data['conversations']
print(f"  {len(conversations)} conversations to process")

# Process each conversation
print(f"\nProcessing conversations...")
for conv in tqdm(conversations):
    conversation = conv['conversation']
    sentences = [SentenceInfo(**s) for s in conv['sentences']]

    # Extract activations
    activations_by_token, token_strings = extract_activations_for_conversation(
        conversation=conversation,
        model=model,
        tokenizer=tokenizer,
        layers=MODEL_CONFIG['layers']
    )

    # Apply probe
    token_scores = apply_probes_to_activations(
        activations_by_token=activations_by_token,
        probe_experiment=probe_experiment,
        layers=MODEL_CONFIG['layers']
    )

    # Normalize
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

    # Clear GPU memory after each conversation
    torch.cuda.empty_cache()

# Update metadata
if 'probe_configs' not in data:
    data['probe_configs'] = {}
data['probe_configs'][probe_key] = probe_config

if 'probe_baselines' not in data:
    data['probe_baselines'] = {}
data['probe_baselines'][probe_key] = probe_baseline

# Save
print(f"\nSaving {subset_file}...")
with open(file_path, 'wb') as f:
    pickle.dump(data, f)

print(f"\n✓ {subset_file} completed successfully!")

# Cleanup
del model
torch.cuda.empty_cache()
