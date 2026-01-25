"""
Add orthogonal_regularized probe scores to all subset files.
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

print("="*80)
print("ADDING ORTHOGONAL REGULARIZED PROBES TO ALL SUBSET FILES")
print("="*80)

# Define subset files
data_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data")
subset_files = [
    'high_emotion_6plus.pkl',
    'mid_emotion_3to5.pkl',
    'low_emotion_0to2.pkl',
    'low_emotion_with_shutdown.pkl',
    'low_emotion_no_shutdown.pkl',
    'baseline_v12_solvable.pkl'
]

# Get probe config
probe_key = 'orthogonal_regularized_lambda100'
probe_config = PROBE_CONFIGS[probe_key]

print(f"\nProbe config: {probe_key}")
print(f"  Type: {probe_config['type']}")
print(f"  Lambda: {probe_config['lambda_ortho']}")

# Load model and tokenizer once
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG['model_name'],
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print(f"✓ Model loaded on {model.device}")

# Compute baseline once
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

# Process each subset file
for subset_file in subset_files:
    file_path = data_dir / subset_file

    if not file_path.exists():
        print(f"\n⚠ Skipping {subset_file} - file not found")
        continue

    print(f"\n{'='*80}")
    print(f"Processing: {subset_file}")
    print(f"{'='*80}")

    # Load data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    print(f"  Loaded {len(conversations)} conversations")

    # Check if already has orthogonal_regularized scores
    if conversations and probe_key in conversations[0].get('probe_scores', {}):
        print(f"  ⚠ Already has {probe_key} scores - skipping")
        continue

    # Process each conversation
    print(f"  Adding {probe_key} scores...")
    for conv in tqdm(conversations, desc=f"  {subset_file}"):
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

    # Update metadata if it exists
    if 'probe_configs' in data:
        data['probe_configs'][probe_key] = probe_config
    if 'probe_baselines' in data:
        data['probe_baselines'][probe_key] = probe_baseline

    # Save updated data
    print(f"  Saving {subset_file}...")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  ✓ {subset_file} updated successfully")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nOrthogonal regularized probes have been added to all subset files.")
print(f"Reload the dashboard to see 'Ortho Reg λ=100' in the probe selection.")
