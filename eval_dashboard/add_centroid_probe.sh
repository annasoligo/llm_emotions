#!/bin/bash
#SBATCH --job-name=add_centroid
#SBATCH --output=/workspace-vast/annas/logs/add_centroid_%j.log
#SBATCH --error=/workspace-vast/annas/logs/add_centroid_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2:00:00

echo "Adding centroid_k10 and orthogonal_cpca_top20 probes to existing data"
echo "Job ID: $SLURM_JOB_ID"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python3 << 'EOF'
import pickle
import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from probes.scripts.token_level_helpers import TokenLevelExperiment
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from eval_dashboard.probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from eval_dashboard.data_preprocessing import extract_activations, apply_probes_to_activations
from eval_dashboard.sentence_aggregator import aggregate_scores_to_sentences, SentenceInfo

print("Loading existing data...")
with open('eval_dashboard/data/preprocessed_conversations.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Current probe types: {list(data['conversations'][0]['probe_scores'].keys())}")

print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG['model_name'],
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# Initialize both probe experiments
probe_configs = {
    'centroid_k10': PROBE_CONFIGS['centroid_k10'],
    'orthogonal_cpca_top20': PROBE_CONFIGS['orthogonal_cpca_top20']
}

probe_experiments = {}
for probe_key, probe_config in probe_configs.items():
    print(f"\nInitializing {probe_key} probe...")
    probe_experiments[probe_key] = TokenLevelExperiment(
        model=None,
        tokenizer=tokenizer,
        probe_type=probe_config['type'],
        probe_dir=probe_config.get('probe_dir'),
        cpca_path=probe_config.get('cpca_path'),
        k_value=probe_config.get('k_value'),
        orthogonality_weight=probe_config.get('orthogonality_weight'),
        orthogonal_representation=probe_config.get('orthogonal_representation'),
        n_components=probe_config.get('n_components'),
        centroid_probe_format=probe_config.get('centroid_probe_format'),
        emotions=EMOTIONS
    )

print(f"\nProcessing {len(data['conversations'])} conversations...")
for i, conv in enumerate(data['conversations']):
    print(f"  [{i+1}/{len(data['conversations'])}]", end='\r')

    conversation = conv['conversation']
    sentences = [SentenceInfo(**s) for s in conv['sentences']]

    # Extract activations (shared for both probes)
    activations_by_token, _ = extract_activations(
        model=model,
        tokenizer=tokenizer,
        conversation=conversation,
        layers=MODEL_CONFIG['layers']
    )

    # Apply both probes
    for probe_key, probe_experiment in probe_experiments.items():
        token_scores = apply_probes_to_activations(
            activations_by_token=activations_by_token,
            probe_experiment=probe_experiment,
            layers=MODEL_CONFIG['layers']
        )

        # Aggregate to sentences
        sentence_scores = aggregate_scores_to_sentences(
            sentences=sentences,
            token_scores=token_scores,
            aggregation='mean'
        )

        # Add to conversation (no baseline normalization for these probes)
        conv['probe_scores'][probe_key] = sentence_scores

print(f"\n✓ Added centroid_k10 and orthogonal_cpca_top20 to all conversations")

print("\nSaving...")
with open('eval_dashboard/data/preprocessed_conversations.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✓ Done!")
EOF

date
