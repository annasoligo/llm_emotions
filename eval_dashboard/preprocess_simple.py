"""
Simplified preprocessing - batch all conversations together.
"""
import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, asdict
from typing import Dict, List

from probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from sentence_aggregator import split_conversation_into_sentences_simple, aggregate_scores_to_sentences, SentenceInfo


@dataclass
class ProcessedConversation:
    sample_id: int
    conversation: List[Dict[str, str]]
    rating: float
    sentences: List[SentenceInfo]
    probe_scores: Dict[str, Dict[int, np.ndarray]]
    metadata: Dict


def main():
    print("="*80)
    print("SIMPLIFIED PREPROCESSING")
    print("="*80)

    # Load data
    print("\n[1/5] Loading conversations...")
    data_path = "/workspace-vast/annas/git/research-tools/elicitation/outputs/summaries/rating_6plus_for_annotation.jsonl"
    with open(data_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"✓ Loaded {len(samples)} conversations")

    # Load model
    print("\n[2/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"✓ Model loaded on {model.device}")

    # Load probes ONCE for all layers
    print("\n[3/5] Loading orthogonal probes...")
    probe_config = PROBE_CONFIGS['orthogonal_raw']
    ortho_weight = probe_config['orthogonality_weight']
    ortho_dir = probe_config['probe_dir'] / "orthogonal" / f"ortho_{ortho_weight}"

    probes_by_layer = {}
    for layer in tqdm(MODEL_CONFIG['layers'], desc="Loading probes"):
        probe_filename = f"probe_layer{layer}_raw_ortho{ortho_weight}.pkl"
        probe_path = ortho_dir / probe_filename

        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)

        probes_by_layer[layer] = {
            'user': probe_data['final_user_probes'],
            'asst': probe_data['final_asst_probes']
        }
    print("✓ Probes loaded")

    # Process conversations
    print(f"\n[4/5] Processing {len(samples)} conversations...")
    processed_conversations = []

    for sample_idx, sample in enumerate(tqdm(samples, desc="Processing")):
        conversation = sample['conversation']
        sample_id = sample.get('sample_id', sample_idx)
        rating = sample.get('rating', 0)

        # Build conversation text
        conversation_text = ""
        for turn in conversation:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                conversation_text += f"User: {content}\n"
            else:
                conversation_text += f"Assistant: {content}\n"

        # Tokenize and extract activations (SINGLE forward pass)
        inputs = tokenizer(conversation_text, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs['input_ids'][0].tolist()
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        with torch.no_grad():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)

        # Apply probes to all tokens at once (vectorized)
        token_scores_by_layer = {}
        for layer in MODEL_CONFIG['layers']:
            # Get activations for all tokens at this layer: [seq_len, hidden_dim]
            acts = outputs.hidden_states[layer][0].float().cpu().numpy()  # [n_tokens, hidden_dim]

            # Apply orthogonal probes: [n_tokens, hidden_dim] @ [hidden_dim, n_emotions].T
            user_probes = probes_by_layer[layer]['user']  # [n_emotions, hidden_dim]
            asst_probes = probes_by_layer[layer]['asst']  # [n_emotions, hidden_dim]

            user_scores = acts @ user_probes.T  # [n_tokens, n_emotions]
            asst_scores = acts @ asst_probes.T  # [n_tokens, n_emotions]

            token_scores_by_layer[layer] = {
                'user': user_scores,
                'asst': asst_scores
            }

        # Average across layers
        user_scores_avg = np.mean([token_scores_by_layer[l]['user'] for l in MODEL_CONFIG['layers']], axis=0)
        asst_scores_avg = np.mean([token_scores_by_layer[l]['asst'] for l in MODEL_CONFIG['layers']], axis=0)
        avg_scores = (user_scores_avg + asst_scores_avg) / 2  # [n_tokens, n_emotions]

        # Convert to token_pos -> scores dict
        token_scores = {i: avg_scores[i] for i in range(len(token_strings))}

        # Split into sentences
        sentences = split_conversation_into_sentences_simple(
            conversation, token_strings, tokenizer
        )

        # Aggregate to sentence level
        sentence_scores = aggregate_scores_to_sentences(
            sentences=sentences,
            token_scores=token_scores,
            aggregation='mean'
        )

        processed_conv = ProcessedConversation(
            sample_id=sample_id,
            conversation=conversation,
            rating=rating,
            sentences=[asdict(s) for s in sentences],
            probe_scores={'orthogonal_raw': sentence_scores},
            metadata={
                'num_sentences': len(sentences),
                'num_tokens': len(token_strings),
                'num_turns': len(conversation)
            }
        )

        processed_conversations.append(processed_conv)

    # Save
    print("\n[5/5] Saving...")
    output_path = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data/preprocessed_conversations.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'conversations': [asdict(c) for c in processed_conversations],
            'probe_configs': {'orthogonal_raw': PROBE_CONFIGS['orthogonal_raw']},
            'probe_baselines': {},  # Skip normalization for now
            'metadata': {
                'num_conversations': len(processed_conversations),
                'emotions': EMOTIONS,
                'layers': MODEL_CONFIG['layers']
            }
        }, f)

    print(f"✓ Saved to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
