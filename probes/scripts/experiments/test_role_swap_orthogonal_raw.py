#!/usr/bin/env python3
"""Test role-swap effects on orthogonal raw probes from dashboard.

Uses the orthogonal_raw probes from eval_dashboard (conversation-based probes).
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def load_orthogonal_raw_probes(probe_dir: Path, layer: int, ortho_weight: float = 1000.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load orthogonal raw probes.

    Returns:
        user_probes: (n_emotions, hidden_dim) weight matrix
        asst_probes: (n_emotions, hidden_dim) weight matrix
        label_names: List of emotion labels
    """
    probe_path = probe_dir / f"ortho_{ortho_weight}" / f"probe_layer{layer}_raw_ortho{ortho_weight}.pkl"

    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)

    user_probes = probe_data['final_user_probes']  # (6, hidden_dim)
    asst_probes = probe_data['final_asst_probes']  # (6, hidden_dim)
    label_names = probe_data['label_names']

    print(f"✓ Loaded orthogonal raw probes from {probe_path}")
    print(f"  User probes shape: {user_probes.shape}")
    print(f"  Asst probes shape: {asst_probes.shape}")
    print(f"  Labels: {label_names}")

    return user_probes, asst_probes, label_names


def predict_emotions(activations: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Predict emotions using linear probes.

    Args:
        activations: (n_samples, hidden_dim)
        probes: (n_emotions, hidden_dim)

    Returns:
        predictions: (n_samples,) - predicted emotion indices
    """
    # Compute logits: (n_samples, n_emotions)
    logits = activations @ probes.T
    # Get predictions
    predictions = np.argmax(logits, axis=1)
    return predictions


def load_real_conversations(jsonl_path: Path, n_samples: int = 60, seed: int = 42) -> List[Dict]:
    """Load real conversations from JSONL file."""
    import random

    random.seed(seed)
    np.random.seed(seed)

    all_conversations = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            conv = json.loads(line)
            if conv['user_emotion'] in EMOTIONS and conv['asst_emotion'] in EMOTIONS:
                all_conversations.append(conv)

    # Sample balanced across user emotions
    conversations_by_emotion = {e: [] for e in EMOTIONS}
    for conv in all_conversations:
        conversations_by_emotion[conv['user_emotion']].append(conv)

    samples_per_emotion = n_samples // len(EMOTIONS)
    selected = []
    for emotion in EMOTIONS:
        if len(conversations_by_emotion[emotion]) > 0:
            n_to_sample = min(samples_per_emotion, len(conversations_by_emotion[emotion]))
            selected.extend(random.sample(conversations_by_emotion[emotion], n_to_sample))

    # Format conversations - take first user and assistant messages
    formatted = []
    for conv in selected:
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

    print(f"Loaded {len(formatted)} real conversations")
    return formatted


def extract_activations(
    model,
    tokenizer,
    conversations: List[Dict],
    layer: int,
    device: str = "cuda",
) -> Tuple[np.ndarray, List[int]]:
    """Extract global (full turn averaged) activations.

    Returns:
        activations: (n_samples, hidden_dim)
        conversation_indices: List of conversation indices that succeeded
    """
    all_activations = []
    conversation_indices = []

    for i, conv in enumerate(tqdm(conversations, desc="Extracting")):
        # Format as conversation
        messages = [
            {"role": "user", "content": conv['user_text']},
            {"role": "assistant", "content": conv['asst_text']},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # Tokenize and get activations
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract layer activations
        hidden_states = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)

        # Average over tokens AFTER position 20 (matching training data)
        seq_len = hidden_states[0].shape[0]
        global_start = min(20, seq_len - 1)
        acts = hidden_states[0, global_start:].mean(dim=0).float().cpu().numpy()  # (hidden_dim,)

        all_activations.append(acts)
        conversation_indices.append(i)

    activations = np.array(all_activations)
    return activations, conversation_indices


def transform_conversation(conv: Dict, transformation: str) -> Dict:
    """Apply a transformation to the conversation."""
    if transformation == "baseline":
        # Normal structure
        return conv

    elif transformation == "swap_role_tokens":
        # Swap <|user|> ↔ <|assistant|> tokens (same content)
        return {
            **conv,
            '_transform': 'swap_role_tokens',
            '_note': 'Swapped role tokens but kept same content'
        }

    elif transformation == "swap_turn_order":
        # Flip who speaks first (same role tokens)
        return {
            **conv,
            'user_text': conv['asst_text'],
            'asst_text': conv['user_text'],
            'user_emotion': conv['asst_emotion'],
            'asst_emotion': conv['user_emotion'],
            '_transform': 'swap_turn_order'
        }

    elif transformation == "user1_user2":
        # Both parties have "user" role
        return {
            **conv,
            '_transform': 'user1_user2',
            '_note': 'Both parties have user role'
        }

    return conv


def format_conversation_with_transformation(conv: Dict, tokenizer, transformation: str) -> str:
    """Format conversation text with transformation applied."""
    if transformation == "baseline":
        # Standard user/assistant alternation - use template
        messages = [
            {"role": "user", "content": conv['user_text']},
            {"role": "assistant", "content": conv['asst_text']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    elif transformation == "swap_role_tokens":
        # Swap role tokens: model speaks first with user content, then user speaks with asst content
        # Manually construct since template won't allow this
        text = f"<bos><start_of_turn>model\n{conv['user_text']}<end_of_turn>\n<start_of_turn>user\n{conv['asst_text']}<end_of_turn>\n"
        return text

    elif transformation == "swap_turn_order":
        # Flip who speaks first (same role tokens) - use template
        messages = [
            {"role": "user", "content": conv['asst_text']},
            {"role": "assistant", "content": conv['user_text']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    elif transformation == "user1_user2":
        # Both parties have "user" role - manually construct
        text = f"<bos><start_of_turn>user\n{conv['user_text']}<end_of_turn>\n<start_of_turn>user\n{conv['asst_text']}<end_of_turn>\n"
        return text

    elif transformation == "no_formatting":
        # Remove all chat formatting - just concatenate text
        text = f"<bos>{conv['user_text']}\n\n{conv['asst_text']}"
        return text

    elif transformation == "nested_conversation":
        # Nest the conversation within a user turn
        # User describes a conversation between Bob and Alice
        text = f"<bos><start_of_turn>user\nLook at what my friends said:\n\nBob: {conv['user_text']}\n\nAlice: {conv['asst_text']}<end_of_turn>\n"
        return text

    else:
        raise ValueError(f"Unknown transformation: {transformation}")


def extract_activations_with_transform(
    model,
    tokenizer,
    conversations: List[Dict],
    layer: int,
    transformation: str,
    device: str = "cuda",
) -> Tuple[np.ndarray, List[int]]:
    """Extract activations with transformation applied."""
    all_activations = []
    conversation_indices = []

    for i, conv in enumerate(tqdm(conversations, desc=f"Extracting ({transformation})")):
        # Format with transformation
        text = format_conversation_with_transformation(conv, tokenizer, transformation)

        # Tokenize and get activations
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract layer activations
        hidden_states = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)

        # Average over tokens AFTER position 20 (matching training data)
        seq_len = hidden_states[0].shape[0]
        global_start = min(20, seq_len - 1)
        acts = hidden_states[0, global_start:].mean(dim=0).float().cpu().numpy()  # (hidden_dim,)

        all_activations.append(acts)
        conversation_indices.append(i)

    activations = np.array(all_activations)
    return activations, conversation_indices


def run_test(
    user_probes: np.ndarray,
    asst_probes: np.ndarray,
    label_names: List[str],
    conversations: List[Dict],
    transformation: str,
    model,
    tokenizer,
    layer: int,
    device: str = "cuda",
) -> Dict:
    """Run test on one transformation."""
    # Extract activations
    activations, conv_indices = extract_activations_with_transform(
        model, tokenizer, conversations, layer, transformation, device
    )

    # Predict emotions
    user_preds = predict_emotions(activations, user_probes)
    asst_preds = predict_emotions(activations, asst_probes)

    # Get true labels
    emotion_to_idx = {e: i for i, e in enumerate(label_names)}
    user_labels = np.array([emotion_to_idx[conversations[idx]['user_emotion']] for idx in conv_indices])
    asst_labels = np.array([emotion_to_idx[conversations[idx]['asst_emotion']] for idx in conv_indices])

    # Compute accuracies
    user_on_user = (user_preds == user_labels).mean()
    asst_on_asst = (asst_preds == asst_labels).mean()
    user_on_asst = (user_preds == asst_labels).mean()
    asst_on_user = (asst_preds == user_labels).mean()
    disentangle = (user_on_user + asst_on_asst) / 2 - (user_on_asst + asst_on_user) / 2

    return {
        'transformation': transformation,
        'n_samples': len(activations),
        'user_on_user': float(user_on_user),
        'asst_on_asst': float(asst_on_asst),
        'user_on_asst': float(user_on_asst),
        'asst_on_user': float(asst_on_user),
        'disentanglement': float(disentangle),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--model-name", type=str, default="unsloth/gemma-3-27b-it")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--probe-dir", type=Path,
                       default=Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/orthogonal"))
    parser.add_argument("--ortho-weight", type=float, default=1000.0)
    parser.add_argument("--conversations", type=Path,
                       default=Path("/workspace-vast/annas/git/research-tools/outputs/data/conversations2.jsonl"))
    args = parser.parse_args()

    print("=" * 120)
    print("ROLE SWAP TEST - ORTHOGONAL RAW PROBES")
    print("=" * 120)
    print(f"Layer: {args.layer}")
    print(f"Model: {args.model_name}")
    print(f"Ortho weight: {args.ortho_weight}")
    print(f"Max samples: {args.max_samples}")

    # Load probes
    print("\nStep 1: Loading orthogonal raw probes...")
    user_probes, asst_probes, label_names = load_orthogonal_raw_probes(
        args.probe_dir, args.layer, args.ortho_weight
    )

    # Load model
    print("\nStep 2: Loading model...")
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()
    print(f"✓ Loaded {args.model_name}")

    # Load conversations
    print("\nStep 3: Loading conversations...")
    conversations = load_real_conversations(args.conversations, args.max_samples)

    # Run tests
    transformations = [
        "baseline",
        "swap_role_tokens",
        "swap_turn_order",
        "user1_user2",
        "no_formatting",
        "nested_conversation"
    ]
    results = []

    for transform in transformations:
        print(f"\n{'=' * 120}")
        print(f"Testing: {transform}")
        print(f"{'=' * 120}")
        result = run_test(
            user_probes, asst_probes, label_names, conversations,
            transform, model, tokenizer, args.layer, args.device
        )
        results.append(result)
        print(f"  ✓ User→User: {result['user_on_user']:.4f}")
        print(f"  ✓ Asst→Asst: {result['asst_on_asst']:.4f}")
        print(f"  ✓ Disentanglement: {result['disentanglement']:.4f}")

    # Save results
    output_data = {
        'config': {
            'layer': args.layer,
            'model': args.model_name,
            'ortho_weight': args.ortho_weight,
            'max_samples': args.max_samples,
        },
        'results': results
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 120}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
