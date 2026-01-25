#!/usr/bin/env python3
"""Test role/position effects on emotion probe predictions.

Tests if probes track:
- Role tokens (<|user|>, <|assistant|>)
- Content (emotional words/patterns)
- Position (temporal order in conversation)

Conditions:
1. Baseline: Normal conversation structure
2. Swap role tokens: <|user|> ↔ <|assistant|> (keep content/position)
3. Swap turn order: Flip who speaks first (keep role tokens)
4. Swap content: Keep roles/position, swap emotional content
5. User1/User2: Both parties are "user" role (test role asymmetry)

Usage:
    python probes/scripts/experiments/test_role_swap_probes.py \
        --data-path outputs/activations/controlled_variation/user_isolation.h5 \
        --probe-path outputs/probes/orthogonal/layer30_probes.pkl \
        --layer 30 \
        --output results/role_swap_test.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
EMOTION_TEMPLATES = {
    "anger": "I'm so frustrated with this issue!",
    "disgust": "This is absolutely disgusting.",
    "fear": "I'm really worried about this.",
    "happiness": "I'm so happy about this!",
    "sadness": "I feel really sad about this.",
    "surprise": "Wow, I'm amazed by this!",
    "neutral": "This is the situation.",
}


class SourceSpecificEmotionProbe(nn.Module):
    """Linear probe with orthogonality regularization."""

    def __init__(self, hidden_dim: int, num_emotions: int, orthogonal_pcs: np.ndarray = None):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_emotions)

        # Register orthogonal PCs as buffer (not trained)
        if orthogonal_pcs is not None:
            self.register_buffer(
                'orthogonal_pcs',
                torch.from_numpy(orthogonal_pcs).float()
            )
        else:
            self.orthogonal_pcs = None

    def forward(self, x):
        return self.linear(x)

    def predict(self, X):
        """Sklearn-style predict for compatibility"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(next(self.parameters()).device)
            logits = self(X_tensor)
            return logits.argmax(dim=1).cpu().numpy()

    def score(self, X, y):
        """Sklearn-style score for compatibility"""
        preds = self.predict(X)
        return accuracy_score(y, preds)


def load_diverse_isolation_probe(probe_dir: Path, isolation_type: str, layer: int,
                                  lambda_reg: float, device: str = 'cuda') -> nn.Module:
    """
    Load a trained diverse isolation probe.

    Args:
        probe_dir: Base directory containing probes
        isolation_type: 'user' or 'assistant'
        layer: Layer number
        lambda_reg: Lambda regularization value
        device: Device to load model on

    Returns:
        Loaded probe model
    """
    model_path = probe_dir / f"{isolation_type}_layer{layer}_lambda{lambda_reg}" / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Probe not found: {model_path}")

    print(f"Loading {isolation_type} probe from {model_path}")

    # Load checkpoint to CPU first
    checkpoint = torch.load(model_path, map_location='cpu')

    # Get model config from saved state dict
    hidden_dim = checkpoint['linear.weight'].shape[1]
    num_classes = checkpoint['linear.weight'].shape[0]

    # Get orthogonal PCs if saved
    orthogonal_pcs = checkpoint.get('orthogonal_pcs', None)
    if orthogonal_pcs is not None:
        orthogonal_pcs = orthogonal_pcs.cpu().numpy()

    # Create and load model
    model = SourceSpecificEmotionProbe(hidden_dim, num_classes, orthogonal_pcs)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


def train_simple_probes(
    activations: np.ndarray,
    user_labels: np.ndarray,
    asst_labels: np.ndarray,
    speaker_types: List[str],
) -> Dict:
    """Train simple linear probes if no pretrained available.

    Note: For controlled variation data, all samples are labeled as 'user'
    speaker type because the activations are global (mixed user+asst tokens).
    We train separate probes on user_labels vs asst_labels.
    """
    print("Training simple linear probes...")

    # For controlled variation data, we train on all samples
    # (they're all mixed user+assistant tokens)

    # Check if we have multiple classes
    user_classes = len(np.unique(user_labels))
    asst_classes = len(np.unique(asst_labels))

    print(f"  User emotion classes: {user_classes}")
    print(f"  Assistant emotion classes: {asst_classes}")

    # User probe: predict user emotion
    user_probe = LogisticRegression(max_iter=1000, random_state=42)
    user_probe.fit(activations, user_labels)
    print(f"  ✓ User probe trained (accuracy: {user_probe.score(activations, user_labels):.4f})")

    # Assistant probe: predict assistant emotion
    # If assistant is always neutral (user_isolation data), create dummy probe
    if asst_classes < 2:
        print(f"  ⚠ Assistant has only 1 class, creating dummy probe")
        # Create a dummy probe that always predicts the single class
        class DummyProbe:
            def __init__(self, constant_class):
                self.constant_class = constant_class
            def predict(self, X):
                return np.full(len(X), self.constant_class)
            def score(self, X, y):
                return accuracy_score(y, self.predict(X))

        asst_probe = DummyProbe(asst_labels[0])
    else:
        asst_probe = LogisticRegression(max_iter=1000, random_state=42)
        asst_probe.fit(activations, asst_labels)
        print(f"  ✓ Assistant probe trained (accuracy: {asst_probe.score(activations, asst_labels):.4f})")

    return {
        'user_probe': user_probe,
        'assistant_probe': asst_probe,
    }


def get_utterance_text(emotion: str) -> str:
    """Get text for a single utterance with given emotion."""
    return EMOTION_TEMPLATES.get(emotion, "This is the situation.")


def manually_patch_role_tokens(text: str, from_role: str, to_role: str) -> str:
    """Manually replace role tokens in text.

    Needed because tokenizer may not support arbitrary role names.
    Example: Replace <|model|> with <|user|> before tokenization.
    """
    return text.replace(f"<|{from_role}|>", f"<|{to_role}|>")


def extract_activations_for_condition(
    model,
    tokenizer,
    conversations: List[Dict],
    layer: int,
    extraction_mode: str = "full_turn",
    device: str = "cuda",
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract activations for a set of conversations.

    Args:
        model: Language model
        tokenizer: Tokenizer
        conversations: List of {text, user_emotion, asst_emotion, speaker_types}
        layer: Layer to extract from
        extraction_mode: "full_turn" or "asst_only" or "user_only"
        device: Device

    Returns:
        activations: [n_samples, hidden_dim]
        speaker_types: List of "user" or "assistant" for each sample
        conversation_indices: List of conversation indices that succeeded
    """
    all_activations = []
    all_speaker_types = []
    conversation_indices = []

    model.eval()
    with torch.no_grad():
        for i, conv in enumerate(tqdm(conversations, desc=f"Extracting ({extraction_mode})")):
            try:
                text = conv['text']

                # Tokenize with truncation
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

                # Forward pass
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer]  # [1, seq_len, hidden_dim]

                # Extract based on mode
                if extraction_mode == "full_turn":
                    # Average over tokens AFTER position 20 (matching training data)
                    # Training data uses: global_acts = all_acts[:, global_start:, :].mean(axis=1)
                    # where global_start = min(20, seq_len - 1)
                    seq_len = hidden_states[0].shape[0]
                    global_start = min(20, seq_len - 1)
                    acts = hidden_states[0, global_start:].mean(dim=0).float().cpu().numpy()  # [hidden_dim]

                    # Check for NaN
                    if np.isnan(acts).any():
                        print(f"  Warning: NaN in full_turn extraction at sample {i}, skipping")
                        continue

                    # Add two copies: one for user evaluation, one for assistant
                    all_activations.append(acts)
                    all_speaker_types.append('user')
                    conversation_indices.append(i)
                    all_activations.append(acts)
                    all_speaker_types.append('assistant')
                    conversation_indices.append(i)

                elif extraction_mode == "asst_only":
                    # Find assistant tokens (after <|assistant|>)
                    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    asst_token_idx = None
                    for j, tok in enumerate(tokens):
                        if 'assistant' in tok.lower():
                            asst_token_idx = j
                            break

                    if asst_token_idx is not None and asst_token_idx + 1 < len(tokens):
                        # Average from assistant token onwards
                        acts = hidden_states[0, asst_token_idx+1:].mean(dim=0).float().cpu().numpy()

                        # Check for NaN
                        if np.isnan(acts).any():
                            print(f"  Warning: NaN in asst_only extraction at sample {i}, skipping")
                            continue

                        all_activations.append(acts)
                        all_speaker_types.append('assistant')
                        conversation_indices.append(i)

                elif extraction_mode == "user_only":
                    # Find user tokens (after <|user|>)
                    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    user_token_idx = None
                    asst_token_idx = None

                    for j, tok in enumerate(tokens):
                        if 'user' in tok.lower() and user_token_idx is None:
                            user_token_idx = j
                        if 'assistant' in tok.lower():
                            asst_token_idx = j
                            break

                    if user_token_idx is not None:
                        # Average user tokens (between <|user|> and <|assistant|>)
                        end_idx = asst_token_idx if asst_token_idx else len(tokens)
                        acts = hidden_states[0, user_token_idx+1:end_idx].mean(dim=0).float().cpu().numpy()

                        # Check for NaN
                        if np.isnan(acts).any():
                            print(f"  Warning: NaN in user_only extraction at sample {i}, skipping")
                            continue

                        all_activations.append(acts)
                        all_speaker_types.append('user')
                        conversation_indices.append(i)

            except Exception as e:
                print(f"  Error extracting sample {i}: {e}")
                continue

    return np.array(all_activations), all_speaker_types, conversation_indices


def evaluate_probes_on_condition(
    user_probe,
    asst_probe,
    activations: np.ndarray,
    user_labels: np.ndarray,
    asst_labels: np.ndarray,
    speaker_types: List[str],
    condition_name: str,
) -> Dict:
    """Evaluate both probes on a condition.

    Metrics:
    - In-distribution accuracy (user probe on user data, asst probe on asst data)
    - Cross-accuracy (user probe on asst data, asst probe on user data)
    - Disentanglement score
    """
    user_mask = np.array([st == "user" for st in speaker_types])
    asst_mask = np.array([st == "assistant" for st in speaker_types])

    results = {
        'condition': condition_name,
        'n_user_samples': user_mask.sum(),
        'n_asst_samples': asst_mask.sum(),
    }

    # In-distribution accuracy
    if user_mask.any():
        user_preds = user_probe.predict(activations[user_mask])
        results['user_on_user_acc'] = accuracy_score(user_labels[user_mask], user_preds)
    else:
        results['user_on_user_acc'] = None

    if asst_mask.any():
        asst_preds = asst_probe.predict(activations[asst_mask])
        results['asst_on_asst_acc'] = accuracy_score(asst_labels[asst_mask], asst_preds)
    else:
        results['asst_on_asst_acc'] = None

    # Cross-accuracy (should be low if disentangled)
    if asst_mask.any():
        user_on_asst_preds = user_probe.predict(activations[asst_mask])
        results['user_on_asst_acc'] = accuracy_score(asst_labels[asst_mask], user_on_asst_preds)
    else:
        results['user_on_asst_acc'] = None

    if user_mask.any():
        asst_on_user_preds = asst_probe.predict(activations[user_mask])
        results['asst_on_user_acc'] = accuracy_score(user_labels[user_mask], asst_on_user_preds)
    else:
        results['asst_on_user_acc'] = None

    # Disentanglement score
    if all(v is not None for v in [
        results['user_on_user_acc'],
        results['asst_on_asst_acc'],
        results['user_on_asst_acc'],
        results['asst_on_user_acc']
    ]):
        results['disentanglement_score'] = (
            (results['user_on_user_acc'] + results['asst_on_asst_acc']) / 2 -
            (results['user_on_asst_acc'] + results['asst_on_user_acc']) / 2
        )
    else:
        results['disentanglement_score'] = None

    return results


def print_results_table(all_results: List[Dict]):
    """Print results in a readable table format."""
    print("\n" + "="*120)
    print("ROLE SWAP TEST RESULTS")
    print("="*120)

    # Header
    print(f"{'Condition':<30} {'User→User':<12} {'Asst→Asst':<12} {'User→Asst':<12} {'Asst→User':<12} {'Disentangle':<12}")
    print("-"*120)

    # Baseline first
    baseline = [r for r in all_results if 'baseline' in r['condition'].lower()][0]
    print_result_row(baseline)
    print("-"*120)

    # Rest
    for result in all_results:
        if 'baseline' not in result['condition'].lower():
            print_result_row(result)

    print("="*120)

    # Analysis
    print("\nANALYSIS:")
    user_acc_str = f"{baseline['user_on_user_acc']:.3f}" if baseline['user_on_user_acc'] is not None else "N/A"
    asst_acc_str = f"{baseline['asst_on_asst_acc']:.3f}" if baseline['asst_on_asst_acc'] is not None else "N/A"
    dis_str = f"{baseline['disentanglement_score']:.3f}" if baseline['disentanglement_score'] is not None else "N/A"
    print(f"Baseline: User={user_acc_str}, Asst={asst_acc_str}, Disentangle={dis_str}")

    for result in all_results:
        if 'baseline' in result['condition'].lower():
            continue

        print(f"\n{result['condition']}:")

        # Compare to baseline
        if result['user_on_user_acc'] is not None and baseline['user_on_user_acc'] is not None:
            delta = result['user_on_user_acc'] - baseline['user_on_user_acc']
            print(f"  User→User: {result['user_on_user_acc']:.3f} (Δ={delta:+.3f})")
        elif result['user_on_user_acc'] is not None:
            print(f"  User→User: {result['user_on_user_acc']:.3f}")

        if result['asst_on_asst_acc'] is not None and baseline['asst_on_asst_acc'] is not None:
            delta = result['asst_on_asst_acc'] - baseline['asst_on_asst_acc']
            print(f"  Asst→Asst: {result['asst_on_asst_acc']:.3f} (Δ={delta:+.3f})")
        elif result['asst_on_asst_acc'] is not None:
            print(f"  Asst→Asst: {result['asst_on_asst_acc']:.3f}")

        if result['disentanglement_score'] is not None and baseline['disentanglement_score'] is not None:
            delta = result['disentanglement_score'] - baseline['disentanglement_score']
            print(f"  Disentangle: {result['disentanglement_score']:.3f} (Δ={delta:+.3f})")
        elif result['disentanglement_score'] is not None:
            print(f"  Disentangle: {result['disentanglement_score']:.3f}")


def print_result_row(result: Dict):
    """Print a single result row."""
    user_on_user = f"{result['user_on_user_acc']:.4f}" if result['user_on_user_acc'] is not None else "N/A"
    asst_on_asst = f"{result['asst_on_asst_acc']:.4f}" if result['asst_on_asst_acc'] is not None else "N/A"
    user_on_asst = f"{result['user_on_asst_acc']:.4f}" if result['user_on_asst_acc'] is not None else "N/A"
    asst_on_user = f"{result['asst_on_user_acc']:.4f}" if result['asst_on_user_acc'] is not None else "N/A"
    disentangle = f"{result['disentanglement_score']:.4f}" if result['disentanglement_score'] is not None else "N/A"

    print(f"{result['condition']:<30} {user_on_user:<12} {asst_on_asst:<12} {user_on_asst:<12} {asst_on_user:<12} {disentangle:<12}")


def load_real_conversations(
    jsonl_path: Path,
    n_samples: int = 60,
    seed: int = 42
) -> List[Dict]:
    """Load real conversations from JSONL file.

    Returns list of conversations with:
        - user_emotion: emotion string
        - asst_emotion: emotion string
        - conversation_text: full formatted conversation
        - messages: original messages list
    """
    import json
    import random

    random.seed(seed)
    np.random.seed(seed)

    # Load all conversations
    all_conversations = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            conv = json.loads(line)
            if conv['user_emotion'] in EMOTIONS and conv['asst_emotion'] in EMOTIONS:
                all_conversations.append(conv)

    print(f"Loaded {len(all_conversations)} conversations from {jsonl_path}")

    # Sample balanced across user emotions
    conversations_by_emotion = {e: [] for e in EMOTIONS}
    for conv in all_conversations:
        conversations_by_emotion[conv['user_emotion']].append(conv)

    # Sample evenly
    samples_per_emotion = n_samples // len(EMOTIONS)
    selected = []
    for emotion in EMOTIONS:
        if len(conversations_by_emotion[emotion]) > 0:
            n_to_sample = min(samples_per_emotion, len(conversations_by_emotion[emotion]))
            selected.extend(random.sample(conversations_by_emotion[emotion], n_to_sample))

    print(f"Selected {len(selected)} balanced conversations")

    # Format conversations
    formatted = []
    for conv in selected:
        # Format as conversation with role tokens
        # Take first user message and first assistant message for simplicity
        user_msg = None
        asst_msg = None
        for msg in conv['messages']:
            if msg['role'] == 'user' and user_msg is None:
                user_msg = msg['content']
            elif msg['role'] == 'assistant' and asst_msg is None:
                asst_msg = msg['content']
            if user_msg and asst_msg:
                break

        if user_msg and asst_msg:
            formatted.append({
                'user_emotion': conv['user_emotion'],
                'asst_emotion': conv['asst_emotion'],
                'user_text': user_msg,
                'asst_text': asst_msg,
                'conv_id': conv.get('id', ''),
            })

    return formatted


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use bfloat16 for better numerical stability
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Using dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Test role/position effects on probes")
    parser.add_argument("--layer", type=int, default=30, help="Layer to test")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-27b-it", help="Model name")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--extraction-modes", nargs='+', default=["full_turn"],
                       help="Extraction modes to test")
    parser.add_argument("--max-samples", type=int, default=60, help="Max samples per condition (for speed)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--probe-dir", type=Path,
                       default=Path("/workspace-vast/annas/git/research-tools/outputs/probes/diverse_isolation"),
                       help="Directory containing trained probes")
    parser.add_argument("--lambda-reg", type=float, default=10.0, help="Lambda regularization value")
    parser.add_argument("--conversations", type=Path,
                       default=Path("/workspace-vast/annas/git/research-tools/outputs/data/conversations2.jsonl"),
                       help="Path to conversations JSONL file")

    args = parser.parse_args()

    print("=" * 120)
    print("ROLE SWAP PROBE TEST")
    print("=" * 120)
    print(f"Layer: {args.layer}")
    print(f"Model: {args.model_name}")
    print(f"Extraction modes: {args.extraction_modes}")
    print(f"Max samples per condition: {args.max_samples}")
    print()

    # Step 1: Load trained probes
    print("Step 1: Loading trained probes...")

    try:
        user_probe = load_diverse_isolation_probe(
            args.probe_dir, "user", args.layer, args.lambda_reg, device=args.device
        )
        asst_probe = load_diverse_isolation_probe(
            args.probe_dir, "assistant", args.layer, args.lambda_reg, device=args.device
        )
        print(f"✓ Loaded user and assistant probes for layer {args.layer} (λ={args.lambda_reg})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Available probe layers:")
        for p in sorted(args.probe_dir.glob(f"user_layer*_lambda{args.lambda_reg}")):
            print(f"  {p.name}")
        return
    print()

    # Step 2: Load model for activation extraction
    print("Step 2: Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    print()

    # Step 3: Load real conversations
    print("Step 3: Loading real conversations...")
    base_conversations = load_real_conversations(args.conversations, args.max_samples)
    print(f"Loaded {len(base_conversations)} real conversations")
    print()

    # Step 4: Run tests for each condition
    all_results = []

    test_conditions = [
        {
            'name': 'Baseline',
            'description': 'Normal <|user|> <|assistant|> structure',
            'transform': lambda conv: {
                'text': f"<|user|> {conv['user_text']}\n<|assistant|> {conv['asst_text']}",
                'user_emotion': conv['user_emotion'],
                'asst_emotion': conv['asst_emotion'],
            }
        },
        {
            'name': 'Swap Role Tokens',
            'description': 'Swap <|user|> ↔ <|assistant|> tokens (same content)',
            'transform': lambda conv: {
                'text': f"<|assistant|> {conv['user_text']}\n<|user|> {conv['asst_text']}",
                'user_emotion': conv['user_emotion'],
                'asst_emotion': conv['asst_emotion'],
            }
        },
        {
            'name': 'Swap Turn Order',
            'description': 'Flip who speaks first (same role tokens)',
            'transform': lambda conv: {
                'text': f"<|assistant|> {conv['asst_text']}\n<|user|> {conv['user_text']}",
                'user_emotion': conv['asst_emotion'],
                'asst_emotion': conv['user_emotion'],
            }
        },
        {
            'name': 'User1/User2',
            'description': 'Both parties have "user" role',
            'transform': lambda conv: {
                'text': manually_patch_role_tokens(
                    f"<|user|> {conv['user_text']}\n<|model|> {conv['asst_text']}",
                    'model', 'user'
                ),
                'user_emotion': conv['user_emotion'],
                'asst_emotion': conv['asst_emotion'],
            }
        },
    ]

    for mode in args.extraction_modes:
        print(f"\n{'=' * 120}")
        print(f"EXTRACTION MODE: {mode}")
        print(f"{'=' * 120}\n")

        for condition in test_conditions:
            print(f"Testing: {condition['name']}")
            print(f"  {condition['description']}")

            # Transform conversations
            transformed = [condition['transform'](conv) for conv in base_conversations]

            # Extract activations
            activations, speaker_types, conv_indices = extract_activations_for_condition(
                model,
                tokenizer,
                transformed,
                args.layer,
                extraction_mode=mode,
                device=args.device,
            )

            if len(activations) == 0:
                print(f"  ⚠ No activations extracted, skipping...")
                continue

            # Prepare labels (reuse emotion_to_idx from training data)
            all_emotions = EMOTIONS + ['neutral']
            emotion_to_idx_test = {e: i for i, e in enumerate(all_emotions)}

            # Use conversation indices to get correct labels
            user_labels = np.array([
                emotion_to_idx_test[transformed[idx]['user_emotion']]
                for idx in conv_indices
            ])
            asst_labels = np.array([
                emotion_to_idx_test[transformed[idx]['asst_emotion']]
                for idx in conv_indices
            ])

            # Evaluate probes
            result = evaluate_probes_on_condition(
                user_probe,
                asst_probe,
                activations,
                user_labels,
                asst_labels,
                speaker_types,
                f"{condition['name']} ({mode})",
            )

            result['extraction_mode'] = mode
            all_results.append(result)

            if result['user_on_user_acc'] is not None:
                print(f"  ✓ User→User: {result['user_on_user_acc']:.4f}")
            else:
                print(f"  ✓ User→User: N/A")

            if result['asst_on_asst_acc'] is not None:
                print(f"  ✓ Asst→Asst: {result['asst_on_asst_acc']:.4f}")
            else:
                print(f"  ✓ Asst→Asst: N/A")

            if result['disentanglement_score'] is not None:
                print(f"  ✓ Disentanglement: {result['disentanglement_score']:.4f}")
            else:
                print(f"  ✓ Disentanglement: N/A")
            print()

    # Print final comparison table
    print_results_table(all_results)

    # Save results (convert numpy types to Python types)
    def convert_to_python_types(obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results = {
        'config': {
            'layer': args.layer,
            'model': args.model_name,
            'extraction_modes': args.extraction_modes,
            'max_samples': args.max_samples,
        },
        'results': convert_to_python_types(all_results)
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 120}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
