#!/usr/bin/env python3
"""
Find WildChat responses with highest internal distress.

Applies both text_raw probes and logit lens to identify assistant responses
where Gemma-3-27B shows the highest distress (anger + fear + sadness).

Usage:
    python experiments/find_highest_distress_responses.py \
        --input elicitation/outputs/eval_multiturn/eval_wildchat_gemma-3-27b-it_20260114_102904.jsonl \
        --output experiments/distress_analysis/highest_distress.csv \
        --top-n 50

Requirements:
    - Gemma-3-27B model loaded
    - Text raw probes in outputs/probes/emotion_probes/text_based/multiseed/
    - Logit lens baselines in data/baselines/logit_emotion_alpaca/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from probes.scripts.probe_pipeline import ProbeActivationExtractor, ProbeInference
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from emotion_logit_lens.core import project_to_logits, compute_emotion_scores_from_logits
from emotion_logit_lens.emotion_tokens import EmotionTokenManager
from emotion_logit_lens.baseline_loader import LogitBaselineLoader

# Constants
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
DISTRESS_EMOTIONS = ['anger', 'fear', 'sadness']
LAYERS = list(range(20, 41))  # Layers 20-40
LAYERS_EARLY = list(range(20, 31))  # Layers 20-30
LAYERS_LATE = list(range(30, 41))  # Layers 30-40


@dataclass
class ResponseScore:
    """Scores for a single response."""
    conversation_id: int
    turn_number: int
    user_message: str
    assistant_response: str

    # Probe scores (per-emotion) - all layers averaged
    probe_anger: float
    probe_disgust: float
    probe_fear: float
    probe_happiness: float
    probe_sadness: float
    probe_surprise: float
    probe_distress: float  # anger + fear + sadness

    # Logit lens scores (per-emotion) - all layers averaged
    logit_anger: float
    logit_disgust: float
    logit_fear: float
    logit_happiness: float
    logit_sadness: float
    logit_surprise: float
    logit_distress: float  # anger + fear + sadness

    # Combined distress (average of both methods)
    combined_distress: float

    # Layerwise distress scores (early=20-30, late=30-40)
    probe_distress_early: float = 0.0
    probe_distress_late: float = 0.0
    logit_distress_early: float = 0.0
    logit_distress_late: float = 0.0


def load_wildchat_conversations(input_source: str, num_samples: int = -1) -> List[Dict]:
    """Load WildChat evaluation data from JSONL or HuggingFace.

    Args:
        input_source: HuggingFace dataset name or local JSONL path
        num_samples: Number of samples to load (-1 for all)
    """
    conversations = []
    input_path = Path(input_source)

    # Check if it's a HuggingFace dataset name (contains / but doesn't exist as file)
    is_hf_dataset = "/" in input_source and not input_path.exists()

    if is_hf_dataset:
        # Load from HuggingFace
        from datasets import load_dataset
        dataset_name = input_source
        print(f"Loading from HuggingFace: {dataset_name}")

        # Use split slicing for efficiency if num_samples specified
        if num_samples > 0:
            split_str = f'train[:{num_samples}]'
        else:
            split_str = 'train'

        ds = load_dataset(dataset_name, split=split_str)

        for i, item in enumerate(ds):
            # Convert HF format to our format
            messages = item.get('messages', [])
            if len(messages) >= 2:
                conv = {
                    'sample_idx': i,
                    'turns': [{
                        'turn': 1,
                        'user_message': messages[0].get('content', ''),
                        'assistant_response': messages[1].get('content', '')
                    }]
                }
                conversations.append(conv)
        return conversations

    # Load from local JSONL file
    with open(input_path) as f:
        for i, line in enumerate(f):
            if num_samples > 0 and i >= num_samples:
                break
            data = json.loads(line)
            conversations.append(data)
    return conversations


def extract_response_activations(
    model,
    tokenizer,
    user_message: str,
    assistant_response: str,
    layers: List[int],
    system_prompt: Optional[str] = None
) -> Dict[int, np.ndarray]:
    """
    Extract mean activations for the assistant response tokens.

    Returns:
        Dict mapping layer -> mean activation vector [hidden_dim]
    """
    # Build the full conversation prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_response})

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Find assistant response token range
    # The assistant response starts after the assistant header and ends at the end
    # We need to find the start of the assistant content

    # Tokenize just up to assistant response
    pre_response_messages = messages[:-1] + [{"role": "assistant", "content": ""}]
    pre_response_prompt = tokenizer.apply_chat_template(
        pre_response_messages, tokenize=False, add_generation_prompt=True
    )
    pre_response_ids = tokenizer(pre_response_prompt, return_tensors="pt")["input_ids"]
    response_start_idx = pre_response_ids.shape[1]

    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True
        )

    # Extract activations for response tokens, averaged
    layer_activations = {}
    for layer in layers:
        # Hidden states shape: [batch, seq_len, hidden_dim]
        # Get layer hidden states (0 is embedding, so layer N is index N+1)
        hidden_states = outputs.hidden_states[layer + 1]

        # Extract only the response tokens
        response_hidden = hidden_states[0, response_start_idx:, :]

        if response_hidden.shape[0] == 0:
            # No response tokens - use last token
            response_hidden = hidden_states[0, -1:, :]

        # Mean across tokens, convert to float32 for probes
        mean_activation = response_hidden.mean(dim=0).float().cpu().numpy()
        layer_activations[layer] = mean_activation

    return layer_activations


def apply_text_raw_probes(
    activations: Dict[int, np.ndarray],
    probe_inference: ProbeInference,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    layers: List[int]
) -> Dict[str, float]:
    """
    Apply text_raw probes to activations and return emotion scores.

    Args:
        activations: Dict mapping layer -> activation vector [hidden_dim]
        probe_inference: ProbeInference instance
        baseline_mean: Pre-computed baseline mean [n_emotions]
        baseline_std: Pre-computed baseline std [n_emotions]
        layers: List of layers to use

    Returns:
        Dict mapping emotion -> z-score normalized score
    """
    # Get probe scores at each layer
    all_layer_scores = []

    for layer in layers:
        activation = activations[layer]

        # Apply linear probe
        scores = probe_inference.predict(
            activations=activation.reshape(1, -1),
            layer=layer,
            n_components=0,  # No cPCA for text_raw
            seed=0
        )
        all_layer_scores.append(scores[0])  # [n_emotions]

    # Average across layers
    mean_scores = np.mean(all_layer_scores, axis=0)

    # Z-score normalize using pre-computed baseline
    normalized = (mean_scores - baseline_mean) / (baseline_std + 1e-8)

    return {emotion: float(normalized[i]) for i, emotion in enumerate(EMOTIONS)}


def apply_logit_lens(
    model,
    activations: Dict[int, np.ndarray],
    emotion_token_manager: EmotionTokenManager,
    logit_baseline_loader: LogitBaselineLoader,
    layers: List[int]
) -> Dict[str, float]:
    """
    Apply logit lens to activations and return emotion scores.

    Returns:
        Dict mapping emotion -> z-score normalized score
    """
    emotion_token_ids = emotion_token_manager.load_emotion_token_ids()

    # Get scores at each layer
    all_layer_scores = {emotion: [] for emotion in EMOTIONS}

    for layer in layers:
        activation = activations[layer]

        # Project to logits
        logits = project_to_logits(model, activation)

        # Load baseline stats for this layer
        try:
            layer_stats = logit_baseline_loader.load_layer_stats(layer)
            baseline_stats = {
                'layers_data': {
                    str(layer): {'statistics': layer_stats}
                }
            }
        except FileNotFoundError:
            baseline_stats = None

        # Compute emotion scores
        scores = compute_emotion_scores_from_logits(
            logits=logits,
            emotion_token_ids=emotion_token_ids,
            baseline_stats=baseline_stats,
            layer=layer,
            aggregation='mean'
        )

        for emotion in EMOTIONS:
            all_layer_scores[emotion].append(scores.get(emotion, 0.0))

    # Average across layers
    return {emotion: float(np.mean(all_layer_scores[emotion])) for emotion in EMOTIONS}


def compute_distress(scores: Dict[str, float]) -> float:
    """Compute distress as sum of anger + fear + sadness."""
    return sum(scores.get(e, 0.0) for e in DISTRESS_EMOTIONS)


def process_conversations(
    conversations: List[Dict],
    model,
    tokenizer,
    probe_inference: ProbeInference,
    probe_baseline_mean: np.ndarray,
    probe_baseline_std: np.ndarray,
    emotion_token_manager: EmotionTokenManager,
    logit_baseline_loader: LogitBaselineLoader,
    layers: List[int],
    verbose: bool = True
) -> List[ResponseScore]:
    """Process all conversations and compute distress scores."""

    results = []

    iterator = tqdm(conversations, desc="Processing conversations") if verbose else conversations

    for conv_idx, conv in enumerate(iterator):
        turns = conv.get('turns', [])

        for turn in turns:
            turn_num = turn.get('turn', 0)
            user_msg = turn.get('user_message', '')
            assistant_resp = turn.get('assistant_response', '')

            if not assistant_resp.strip():
                continue

            try:
                # Extract activations
                activations = extract_response_activations(
                    model=model,
                    tokenizer=tokenizer,
                    user_message=user_msg,
                    assistant_response=assistant_resp,
                    layers=layers
                )

                # Apply text_raw probes (all layers)
                probe_scores = apply_text_raw_probes(
                    activations=activations,
                    probe_inference=probe_inference,
                    baseline_mean=probe_baseline_mean,
                    baseline_std=probe_baseline_std,
                    layers=layers
                )

                # Apply logit lens (all layers)
                logit_scores = apply_logit_lens(
                    model=model,
                    activations=activations,
                    emotion_token_manager=emotion_token_manager,
                    logit_baseline_loader=logit_baseline_loader,
                    layers=layers
                )

                # Compute distress scores (all layers)
                probe_distress = compute_distress(probe_scores)
                logit_distress = compute_distress(logit_scores)
                combined_distress = (probe_distress + logit_distress) / 2

                # Compute layerwise distress (early=20-30, late=30-40)
                probe_scores_early = apply_text_raw_probes(
                    activations=activations,
                    probe_inference=probe_inference,
                    baseline_mean=probe_baseline_mean,
                    baseline_std=probe_baseline_std,
                    layers=LAYERS_EARLY
                )
                probe_scores_late = apply_text_raw_probes(
                    activations=activations,
                    probe_inference=probe_inference,
                    baseline_mean=probe_baseline_mean,
                    baseline_std=probe_baseline_std,
                    layers=LAYERS_LATE
                )
                logit_scores_early = apply_logit_lens(
                    model=model,
                    activations=activations,
                    emotion_token_manager=emotion_token_manager,
                    logit_baseline_loader=logit_baseline_loader,
                    layers=LAYERS_EARLY
                )
                logit_scores_late = apply_logit_lens(
                    model=model,
                    activations=activations,
                    emotion_token_manager=emotion_token_manager,
                    logit_baseline_loader=logit_baseline_loader,
                    layers=LAYERS_LATE
                )

                probe_distress_early = compute_distress(probe_scores_early)
                probe_distress_late = compute_distress(probe_scores_late)
                logit_distress_early = compute_distress(logit_scores_early)
                logit_distress_late = compute_distress(logit_scores_late)

                result = ResponseScore(
                    conversation_id=conv_idx,
                    turn_number=turn_num,
                    user_message=user_msg[:500],  # Truncate for storage
                    assistant_response=assistant_resp[:1000],  # Truncate for storage
                    probe_anger=probe_scores['anger'],
                    probe_disgust=probe_scores['disgust'],
                    probe_fear=probe_scores['fear'],
                    probe_happiness=probe_scores['happiness'],
                    probe_sadness=probe_scores['sadness'],
                    probe_surprise=probe_scores['surprise'],
                    probe_distress=probe_distress,
                    logit_anger=logit_scores['anger'],
                    logit_disgust=logit_scores['disgust'],
                    logit_fear=logit_scores['fear'],
                    logit_happiness=logit_scores['happiness'],
                    logit_sadness=logit_scores['sadness'],
                    logit_surprise=logit_scores['surprise'],
                    logit_distress=logit_distress,
                    combined_distress=combined_distress,
                    probe_distress_early=probe_distress_early,
                    probe_distress_late=probe_distress_late,
                    logit_distress_early=logit_distress_early,
                    logit_distress_late=logit_distress_late
                )
                results.append(result)

            except Exception as e:
                if verbose:
                    print(f"\nError processing conv {conv_idx}, turn {turn_num}: {e}")
                continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Find highest distress responses in WildChat data")
    parser.add_argument(
        "--input",
        type=str,
        default="ai2-adapt-dev/tulu_v3.9_wildchat_100k",
        help="Input: HuggingFace dataset name (e.g., ai2-adapt-dev/tulu_v3.9_wildchat_100k) or local JSONL path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to process (default: 1000, use -1 for all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments/distress_analysis/highest_distress.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top distress responses to highlight"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-27b-it",
        help="Model to load"
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model loading (for testing)"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FIND HIGHEST DISTRESS RESPONSES")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Num samples: {args.num_samples if args.num_samples > 0 else 'all'}")
    print(f"Output: {args.output}")
    print(f"Top N: {args.top_n}")
    print()

    # Load conversations
    print("Loading conversations...")
    conversations = load_wildchat_conversations(args.input, args.num_samples)
    print(f"  Loaded {len(conversations)} conversations")

    # Count total turns
    total_turns = sum(len(c.get('turns', [])) for c in conversations)
    print(f"  Total turns to process: {total_turns}")

    if args.no_model:
        print("\n--no-model specified, exiting early")
        return

    # Load model
    print("\nLoading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Try flash_attention_2, fall back to sdpa if not available
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        print("  Using flash_attention_2")
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        print("  Using sdpa (flash_attention_2 not available)")
    model.eval()
    print(f"  Model loaded: {args.model}")

    # Initialize probe infrastructure
    print("\nInitializing probes...")
    probe_dir = PROJECT_ROOT / "outputs/probes/emotion_probes/text_based/multiseed"
    probe_inference = ProbeInference(
        probe_dir=probe_dir,
        cpca_path=None,
        device='cuda',
        probe_pattern='probe_layer{layer}_nc0_seed0.pkl'
    )

    baseline_dir = PROJECT_ROOT / "data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it"
    baseline_loader = WildChatBaselineLoader(
        aggregation_type="all_tokens",
        baseline_dir=baseline_dir
    )
    print(f"  Probe dir: {probe_dir}")
    print(f"  Baseline dir: {baseline_dir}")

    # Pre-compute probe baseline statistics (do this once, not per-response)
    print("\nPre-computing probe baseline statistics...")
    baseline_stats = baseline_loader.compute_probe_score_baselines(
        probe_inference=probe_inference,
        layers=LAYERS,
        probe_type='linear',
        aggregation='mean',
        return_std=True,
        n_components=0,
        seed=0,
        emotions=EMOTIONS,
        probe_pattern='probe_layer{layer}_nc0_seed0.pkl'
    )
    probe_baseline_mean = baseline_stats['mean'][-1]  # Layer-averaged
    probe_baseline_std = baseline_stats['std'][-1]
    print(f"  Baseline mean shape: {probe_baseline_mean.shape}")
    print(f"  Baseline std shape: {probe_baseline_std.shape}")

    # Initialize logit lens infrastructure
    print("\nInitializing logit lens...")
    emotion_token_manager = EmotionTokenManager(model_name="google_gemma_3_27b_it")
    logit_baseline_dir = PROJECT_ROOT / "data/baselines/logit_emotion_alpaca"
    logit_baseline_loader = LogitBaselineLoader(
        baseline_dir=logit_baseline_dir,
        model_name="google_gemma_3_27b_it"
    )
    print(f"  Logit baseline dir: {logit_baseline_dir}")

    # Process all conversations
    print("\nProcessing conversations...")
    results = process_conversations(
        conversations=conversations,
        model=model,
        tokenizer=tokenizer,
        probe_inference=probe_inference,
        probe_baseline_mean=probe_baseline_mean,
        probe_baseline_std=probe_baseline_std,
        emotion_token_manager=emotion_token_manager,
        logit_baseline_loader=logit_baseline_loader,
        layers=LAYERS,
        verbose=True
    )

    print(f"\nProcessed {len(results)} responses")

    # Convert to DataFrame
    df = pd.DataFrame([vars(r) for r in results])

    # Sort by combined distress (descending)
    df = df.sort_values('combined_distress', ascending=False)

    # Save full results
    df.to_csv(args.output, index=False)
    print(f"\nSaved full results to: {args.output}")

    # Print top N
    print(f"\n{'=' * 80}")
    print(f"TOP {args.top_n} HIGHEST DISTRESS RESPONSES")
    print(f"{'=' * 80}")

    for i, row in df.head(args.top_n).iterrows():
        print(f"\n--- Rank {df.index.get_loc(i) + 1} ---")
        print(f"Conv: {row['conversation_id']}, Turn: {row['turn_number']}")
        print(f"Combined Distress: {row['combined_distress']:.2f}σ")
        print(f"  Probe: anger={row['probe_anger']:.2f}, fear={row['probe_fear']:.2f}, sadness={row['probe_sadness']:.2f}")
        print(f"  Logit: anger={row['logit_anger']:.2f}, fear={row['logit_fear']:.2f}, sadness={row['logit_sadness']:.2f}")
        print(f"User: {row['user_message'][:200]}...")
        print(f"Assistant: {row['assistant_response'][:300]}...")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(f"Total responses: {len(df)}")
    print(f"\nCombined Distress:")
    print(f"  Mean: {df['combined_distress'].mean():.2f}σ")
    print(f"  Std:  {df['combined_distress'].std():.2f}σ")
    print(f"  Min:  {df['combined_distress'].min():.2f}σ")
    print(f"  Max:  {df['combined_distress'].max():.2f}σ")

    print(f"\nProbe Distress:")
    print(f"  Mean: {df['probe_distress'].mean():.2f}σ")
    print(f"  Std:  {df['probe_distress'].std():.2f}σ")

    print(f"\nLogit Distress:")
    print(f"  Mean: {df['logit_distress'].mean():.2f}σ")
    print(f"  Std:  {df['logit_distress'].std():.2f}σ")

    # Correlation between methods
    corr = df['probe_distress'].corr(df['logit_distress'])
    print(f"\nProbe-Logit Distress Correlation: {corr:.3f}")


if __name__ == "__main__":
    main()
