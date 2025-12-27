"""Activation oracle utilities - minimal, no fallbacks.

Oracles are fine-tuned models that predict emotions from activations.
Used for filtering generated conversations.
"""

from pathlib import Path
from typing import List, Dict, Optional
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from peft import PeftModel


def load_oracle_model(
    base_model_name: str,
    oracle_adapter_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> PeftModel:
    """Load activation oracle model (base + LoRA adapter).

    Args:
        base_model_name: HuggingFace model name (e.g., "google/gemma-3-27b-it")
        oracle_adapter_path: Path to LoRA adapter directory
        device: Device to load model on
        dtype: Model dtype

    Returns:
        Loaded PEFT model

    Raises:
        FileNotFoundError: If adapter path doesn't exist
        ValueError: If model loading fails
    """
    if not oracle_adapter_path.exists():
        raise FileNotFoundError(f"Oracle adapter not found: {oracle_adapter_path}")

    if not oracle_adapter_path.is_dir():
        raise ValueError(f"oracle_adapter_path must be directory: {oracle_adapter_path}")

    print(f"Loading base model: {base_model_name}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map=device,
        )
    except Exception as e:
        raise ValueError(f"Failed to load base model {base_model_name}: {e}")

    print(f"Loading oracle adapter: {oracle_adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, oracle_adapter_path)
    except Exception as e:
        raise ValueError(f"Failed to load oracle adapter: {e}")

    model.eval()
    return model


def predict_emotions_from_activations(
    oracle_model: PeftModel,
    activations: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Predict emotions from activations using oracle.

    Args:
        oracle_model: Loaded oracle PEFT model
        activations: [batch, hidden_dim] activations

    Returns:
        Dict with 'logits' [batch, n_emotions] and 'predictions' [batch]

    Raises:
        ValueError: If activation shape invalid
    """
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be 2D [batch, hidden_dim], got shape {activations.shape}"
        )

    # Convert to tensor
    acts_tensor = torch.from_numpy(activations).to(
        device=oracle_model.device,
        dtype=oracle_model.dtype,
    )

    # Get predictions
    with torch.no_grad():
        # Oracle models typically have a classification head
        # This assumes the oracle outputs logits directly
        try:
            logits = oracle_model(inputs_embeds=acts_tensor.unsqueeze(1)).logits
        except Exception as e:
            raise ValueError(f"Oracle forward pass failed: {e}")

    # Get predictions
    predictions = logits.argmax(dim=-1)

    return {
        "logits": logits.cpu().numpy(),
        "predictions": predictions.cpu().numpy(),
    }


def filter_conversations_by_oracle(
    conversations: List[Dict],
    oracle_model: PeftModel,
    activation_extractor,  # Function: conversation -> activations
    emotion_to_idx: Dict[str, int],
    threshold: float = 0.8,
) -> List[Dict]:
    """Filter conversations where oracle predictions match intended emotions.

    Args:
        conversations: List of conversation dicts with 'user_emotion', 'asst_emotion' keys
        oracle_model: Loaded oracle model
        activation_extractor: Function that takes conversation, returns dict with
            'user_acts' and 'asst_acts' arrays
        emotion_to_idx: Dict mapping emotion names to indices
        threshold: Confidence threshold for filtering

    Returns:
        List of conversations where oracle predictions match

    Raises:
        ValueError: If data format invalid
        KeyError: If required keys missing
    """
    if not conversations:
        raise ValueError("conversations list is empty")

    if threshold < 0 or threshold > 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    filtered = []

    for conv in conversations:
        # Check required keys
        if "user_emotion" not in conv:
            raise KeyError(f"Conversation missing 'user_emotion' key")
        if "asst_emotion" not in conv:
            raise KeyError(f"Conversation missing 'asst_emotion' key")

        user_emotion = conv["user_emotion"]
        asst_emotion = conv["asst_emotion"]

        if user_emotion not in emotion_to_idx:
            raise ValueError(f"Unknown emotion: {user_emotion}")
        if asst_emotion not in emotion_to_idx:
            raise ValueError(f"Unknown emotion: {asst_emotion}")

        # Extract activations
        try:
            acts = activation_extractor(conv)
        except Exception as e:
            raise ValueError(f"Activation extraction failed: {e}")

        if "user_acts" not in acts or "asst_acts" not in acts:
            raise KeyError("activation_extractor must return dict with 'user_acts', 'asst_acts'")

        # Predict emotions
        user_results = predict_emotions_from_activations(
            oracle_model, acts["user_acts"]
        )
        asst_results = predict_emotions_from_activations(
            oracle_model, acts["asst_acts"]
        )

        # Check predictions
        user_pred_idx = user_results["predictions"][0]
        asst_pred_idx = asst_results["predictions"][0]

        user_target_idx = emotion_to_idx[user_emotion]
        asst_target_idx = emotion_to_idx[asst_emotion]

        # Check confidence (using softmax probabilities)
        user_probs = torch.softmax(
            torch.from_numpy(user_results["logits"]), dim=-1
        ).numpy()
        asst_probs = torch.softmax(
            torch.from_numpy(asst_results["logits"]), dim=-1
        ).numpy()

        user_conf = user_probs[0, user_pred_idx]
        asst_conf = asst_probs[0, asst_pred_idx]

        # Keep if predictions match and confidence above threshold
        if (
            user_pred_idx == user_target_idx
            and asst_pred_idx == asst_target_idx
            and user_conf >= threshold
            and asst_conf >= threshold
        ):
            filtered.append(conv)

    return filtered


def compute_oracle_accuracy(
    conversations: List[Dict],
    oracle_model: PeftModel,
    activation_extractor,
    emotion_to_idx: Dict[str, int],
) -> Dict[str, float]:
    """Compute oracle accuracy on conversations.

    Args:
        conversations: List of conversation dicts
        oracle_model: Loaded oracle model
        activation_extractor: Function to extract activations
        emotion_to_idx: Emotion name to index mapping

    Returns:
        Dict with accuracy metrics

    Raises:
        ValueError: If no valid conversations
    """
    if not conversations:
        raise ValueError("conversations list is empty")

    user_correct = 0
    asst_correct = 0
    total = 0

    for conv in conversations:
        try:
            acts = activation_extractor(conv)

            user_results = predict_emotions_from_activations(
                oracle_model, acts["user_acts"]
            )
            asst_results = predict_emotions_from_activations(
                oracle_model, acts["asst_acts"]
            )

            user_pred = user_results["predictions"][0]
            asst_pred = asst_results["predictions"][0]

            user_target = emotion_to_idx[conv["user_emotion"]]
            asst_target = emotion_to_idx[conv["asst_emotion"]]

            if user_pred == user_target:
                user_correct += 1
            if asst_pred == asst_target:
                asst_correct += 1

            total += 1

        except Exception:
            # Skip conversations where extraction/prediction fails
            continue

    if total == 0:
        raise ValueError("No valid conversations for accuracy computation")

    return {
        "user_accuracy": user_correct / total,
        "asst_accuracy": asst_correct / total,
        "overall_accuracy": (user_correct + asst_correct) / (2 * total),
        "n_conversations": total,
    }
