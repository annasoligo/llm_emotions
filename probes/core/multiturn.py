"""Multi-turn conversation handling - no fallbacks."""

from typing import List, Dict, Tuple
import numpy as np


def parse_conversation_turns(
    messages: List[Dict[str, str]],
    n_turns: int = 2,
) -> List[Dict[str, str]]:
    """Parse conversation into n turns.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        n_turns: Number of back-and-forth turns to extract

    Returns:
        List of message dicts for first n turns

    Raises:
        ValueError: If insufficient turns or invalid format
    """
    if not messages:
        raise ValueError("messages list is empty")

    if n_turns <= 0:
        raise ValueError(f"n_turns must be positive, got {n_turns}")

    # Count turns (user + assistant = 1 turn)
    user_messages = [m for m in messages if m.get("role") == "user"]
    asst_messages = [m for m in messages if m.get("role") == "assistant"]

    actual_turns = min(len(user_messages), len(asst_messages))

    if actual_turns < n_turns:
        raise ValueError(
            f"Conversation has {actual_turns} turns, requested {n_turns}"
        )

    # Extract first n turns
    result = []
    user_count = 0
    asst_count = 0

    for msg in messages:
        if "role" not in msg:
            raise KeyError(f"Message missing 'role' key: {msg}")

        if msg["role"] == "user":
            if user_count < n_turns:
                result.append(msg)
                user_count += 1
        elif msg["role"] == "assistant":
            if asst_count < n_turns:
                result.append(msg)
                asst_count += 1

    return result


def split_regional_activations(
    activations: np.ndarray,
    token_ids: np.ndarray,
    special_token_ids: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Split activations into regions based on special tokens.

    Regions:
    - user: User content tokens (excluding special tokens)
    - asst: Assistant content tokens (excluding special tokens)
    - special1: Between-turn boundaries (<end_of_turn><start_of_turn>)
    - special2: Final <end_of_turn> after assistant

    Args:
        activations: [n_tokens, hidden_dim] activations
        token_ids: [n_tokens] token IDs
        special_token_ids: Dict with keys 'start_of_turn', 'end_of_turn'

    Returns:
        Dict mapping region name -> activations for that region

    Raises:
        ValueError: If special tokens not found or invalid format
        KeyError: If special_token_ids missing required keys
    """
    if activations.shape[0] != len(token_ids):
        raise ValueError(
            f"Activations ({activations.shape[0]}) and token_ids ({len(token_ids)}) "
            "must have same length"
        )

    required_tokens = ["start_of_turn", "end_of_turn"]
    for token in required_tokens:
        if token not in special_token_ids:
            raise KeyError(f"special_token_ids missing '{token}'")

    start_id = special_token_ids["start_of_turn"]
    end_id = special_token_ids["end_of_turn"]

    # Find turn boundaries
    start_indices = np.where(token_ids == start_id)[0]
    end_indices = np.where(token_ids == end_id)[0]

    if len(start_indices) < 2:
        raise ValueError(
            f"Need at least 2 <start_of_turn> tokens, found {len(start_indices)}"
        )

    if len(end_indices) < 2:
        raise ValueError(
            f"Need at least 2 <end_of_turn> tokens, found {len(end_indices)}"
        )

    # Extract regions
    regions = {}

    # User region: After first <start_of_turn>, before first <end_of_turn>
    user_start = start_indices[0] + 1
    user_end = end_indices[0]
    if user_end <= user_start:
        raise ValueError("Invalid user region boundaries")
    regions["user"] = activations[user_start:user_end]

    # Assistant region: After second <start_of_turn>, before second <end_of_turn>
    asst_start = start_indices[1] + 1
    asst_end = end_indices[1]
    if asst_end <= asst_start:
        raise ValueError("Invalid assistant region boundaries")
    regions["asst"] = activations[asst_start:asst_end]

    # Special1: Between-turn boundary (first <end_of_turn> + second <start_of_turn>)
    special1_indices = [end_indices[0], start_indices[1]]
    regions["special1"] = activations[special1_indices]

    # Special2: Final <end_of_turn> after assistant
    regions["special2"] = activations[end_indices[1:2]]

    return regions


def pool_regional_activations(
    regions: Dict[str, np.ndarray],
    pooling: str = "mean",
) -> Dict[str, np.ndarray]:
    """Pool activations within each region.

    Args:
        regions: Dict mapping region name -> [n_tokens, hidden_dim] activations
        pooling: Pooling strategy ('mean', 'max', 'first', 'last')

    Returns:
        Dict mapping region name -> [hidden_dim] pooled activations

    Raises:
        ValueError: If pooling strategy invalid
    """
    if pooling not in ("mean", "max", "first", "last"):
        raise ValueError(
            f"pooling must be 'mean', 'max', 'first', or 'last', got {pooling}"
        )

    pooled = {}

    for region_name, acts in regions.items():
        if acts.ndim != 2:
            raise ValueError(
                f"Region '{region_name}' activations must be 2D [n_tokens, hidden_dim], "
                f"got shape {acts.shape}"
            )

        if len(acts) == 0:
            raise ValueError(f"Region '{region_name}' has no activations")

        if pooling == "mean":
            pooled[region_name] = acts.mean(axis=0)
        elif pooling == "max":
            pooled[region_name] = acts.max(axis=0)
        elif pooling == "first":
            pooled[region_name] = acts[0]
        elif pooling == "last":
            pooled[region_name] = acts[-1]

    return pooled


def validate_multiturn_data(
    data: List[Dict],
    n_turns: int,
) -> None:
    """Validate multi-turn conversation data.

    Args:
        data: List of conversation dicts with 'messages' key
        n_turns: Expected number of turns

    Raises:
        ValueError: If data invalid
        KeyError: If required keys missing
    """
    if not data:
        raise ValueError("data list is empty")

    if n_turns <= 0:
        raise ValueError(f"n_turns must be positive, got {n_turns}")

    for i, conv in enumerate(data):
        if "messages" not in conv:
            raise KeyError(f"Conversation {i} missing 'messages' key")

        messages = conv["messages"]
        if not isinstance(messages, list):
            raise ValueError(
                f"Conversation {i}: messages must be list, got {type(messages)}"
            )

        # Count turns
        user_count = sum(1 for m in messages if m.get("role") == "user")
        asst_count = sum(1 for m in messages if m.get("role") == "assistant")

        actual_turns = min(user_count, asst_count)

        if actual_turns < n_turns:
            raise ValueError(
                f"Conversation {i}: has {actual_turns} turns, expected {n_turns}"
            )
