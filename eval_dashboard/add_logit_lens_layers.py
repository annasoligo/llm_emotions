"""Add logit_lens scores for specific layer range to pickle files."""
import pickle
import numpy as np
from pathlib import Path
import sys
import argparse

# Add emo_lens to path
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")

from emotion_evals.emo_lens.token_trajectories import (
    extract_token_level_activations,
    compute_layer_averaged_baseline_stats,
    compute_token_emotion_scores
)
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_emotion_token_ids_from_json
)

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input pickle file path')
    parser.add_argument('--layer_start', type=int, required=True, help='Start layer (inclusive)')
    parser.add_argument('--layer_end', type=int, required=True, help='End layer (inclusive)')
    args = parser.parse_args()

    input_path = Path(args.input_file)
    layer_start = args.layer_start
    layer_end = args.layer_end
    layer_suffix = f"_l{layer_start}_{layer_end}"

    print("="*80)
    print(f"ADDING LOGIT_LENS (LAYERS {layer_start}-{layer_end}) TO: {input_path.name}")
    print("="*80)

    # Load data
    print(f"\nLoading data from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    num_conversations = len(data['conversations'])
    print(f"✓ Loaded {num_conversations} conversations")

    # Load model using emo_lens loader
    print("\nLoading model...")
    BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
    model, tokenizer = load_base_model(BASE_MODEL_NAME)
    print("✓ Model loaded")

    # Load emotion token IDs
    print("Loading emotion token IDs...")
    emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
    print(f"✓ Loaded {len(emotion_token_ids)} emotions")

    # Load reference statistics
    print("Loading baseline statistics...")
    ACTIVATION_STRATEGY = "generated_tokens_avg"
    SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
    ref_stats = load_reference_stats_for_strategy(
        model_name=BASE_MODEL_NAME,
        activation_strategy=ACTIVATION_STRATEGY,
        script_dir=SCRIPT_DIR
    )
    print("✓ Loaded baseline stats")

    # Compute layer-averaged baseline stats for specified layers
    LAYER_RANGE = list(range(layer_start, layer_end + 1))
    print(f"Using layers: {LAYER_RANGE}")
    averaged_baseline_stats = compute_layer_averaged_baseline_stats(
        ref_stats=ref_stats,
        layers=LAYER_RANGE,
        emotion_token_ids=emotion_token_ids
    )
    print("✓ Computed averaged baseline statistics")

    # Process all conversations
    print(f"\nProcessing {num_conversations} conversations...")

    for i, conv in enumerate(data['conversations']):
        print(f"\n[{i+1}/{num_conversations}] Processing conversation {conv['sample_id']}...")

        # Build conversation text
        conversation_text = ""
        for turn in conv['conversation']:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                conversation_text += f"User: {content}\n"
            else:
                conversation_text += f"Assistant: {content}\n"

        print(f"  Conversation: {len(conversation_text)} chars, {len(conv['sentences'])} sentences")

        # Extract token-level activations
        try:
            activations_by_token, token_ids = extract_token_level_activations(
                model=model,
                tokenizer=tokenizer,
                prompt=conversation_text,
                layers=LAYER_RANGE,
                start_token_idx=0,
                system_prompt=None,
                num_generated_tokens=0
            )
            print(f"  ✓ Extracted {len(activations_by_token)} token activations")
        except Exception as e:
            print(f"  ✗ Error extracting activations: {e}")
            continue

        # Process both mean and max aggregations
        for agg_method in ['mean', 'max']:
            try:
                # Compute emotion scores
                token_emotion_scores = compute_token_emotion_scores(
                    model=model,
                    activations_by_token=activations_by_token,
                    layers=LAYER_RANGE,
                    emotion_token_ids=emotion_token_ids,
                    averaged_baseline_stats=averaged_baseline_stats,
                    aggregation=agg_method,
                    subtract_mean=True
                )
                print(f"  ✓ Computed {agg_method} scores for {len(token_emotion_scores)} tokens")

                # Aggregate to sentence level
                sentence_scores = {}
                for sent in conv['sentences']:
                    sent_id = sent['sentence_id']
                    start_tok = sent['start_token']
                    end_tok = sent['end_token']

                    sent_score_list = []
                    for t in range(start_tok, end_tok):
                        if t in token_emotion_scores:
                            scores_array = np.array([token_emotion_scores[t][e] for e in EMOTIONS])
                            sent_score_list.append(scores_array)

                    if sent_score_list:
                        sentence_scores[sent_id] = np.mean(sent_score_list, axis=0)

                print(f"  ✓ Aggregated to {len(sentence_scores)} sentences ({agg_method})")

                # Store results with layer-specific key
                if 'probe_scores' not in conv:
                    conv['probe_scores'] = {}

                probe_key = f"logit_lens_{agg_method}{layer_suffix}"
                conv['probe_scores'][probe_key] = sentence_scores

            except Exception as e:
                print(f"  ✗ Error computing {agg_method} aggregation: {e}")

    # Save results
    print(f"\nSaving results to {input_path}...")
    with open(input_path, 'wb') as f:
        pickle.dump(data, f)
    print("✓ Saved")

    # Compute statistics
    print("\n" + "="*80)
    print(f"STATISTICS (LAYERS {layer_start}-{layer_end})")
    print("="*80)

    probe_key = f"logit_lens_mean{layer_suffix}"
    all_scores = []
    for conv in data['conversations']:
        if 'probe_scores' in conv and probe_key in conv['probe_scores']:
            for sent_id, scores in conv['probe_scores'][probe_key].items():
                if isinstance(scores, np.ndarray):
                    all_scores.append(scores)

    if all_scores:
        scores_matrix = np.array(all_scores)
        corr_matrix = np.corrcoef(scores_matrix.T)

        off_diag_corrs = []
        for i in range(6):
            for j in range(i+1, 6):
                off_diag_corrs.append(abs(corr_matrix[i, j]))

        print(f"\nProcessed {len(all_scores)} sentences across {num_conversations} conversations")
        print(f"Score statistics:")
        print(f"  Mean: {scores_matrix.mean():.3f}")
        print(f"  Std:  {scores_matrix.std():.3f}")
        print(f"  Min:  {scores_matrix.min():.3f}")
        print(f"  Max:  {scores_matrix.max():.3f}")
        print(f"\nAverage absolute correlation: {np.mean(off_diag_corrs):.3f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
