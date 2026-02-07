"""
Shared utilities for extracting and processing logprobs from vLLM outputs.

This module provides functions for:
- Extracting letter logprobs (A-E) from vLLM outputs
- Computing expected scores from logprobs
- Computing coherence (probability mass on valid letters)

These functions are used by behavioral_shift.py, debug_235b_outputs.py,
and other analysis scripts.
"""

import math
from typing import Dict, Optional, Tuple

from .config import SCORE_MAP, SCORE_MAP_REVERSED, VALID_TOKENS


def extract_letter_logprobs(output, tokenizer=None) -> Dict[str, float]:
    """
    Extract logprobs for A-E tokens from vLLM output, scanning all generated tokens.

    Models often generate newlines before the answer (e.g., '\\n\\nD'), so we need to
    find the first token position that contains a letter A-E and use those logprobs.

    Args:
        output: vLLM RequestOutput object
        tokenizer: Optional tokenizer for decoding token IDs (used as fallback)

    Returns:
        Dict mapping 'A'-'E' to log probabilities
    """
    if not output.outputs[0].logprobs:
        return {}

    # Scan through each token position's logprobs
    for pos_logprobs in output.outputs[0].logprobs:
        letter_logprobs = {}

        for token_id, logprob_obj in pos_logprobs.items():
            # Handle different vLLM versions
            if hasattr(logprob_obj, 'decoded_token'):
                token_str = logprob_obj.decoded_token.strip()
                logprob = logprob_obj.logprob
            elif tokenizer is not None:
                token_str = tokenizer.decode([token_id]).strip()
                logprob = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else logprob_obj
            else:
                token_str = str(token_id).strip()
                logprob = logprob_obj

            # Check if this is a valid response token
            # Handle variations like "A", " A", "(A)", "(A", "a", "A.", etc.
            # Note: Some tokenizers split "(A)" as "(A" + ")" so we check for "(A" too
            for letter in VALID_TOKENS:
                letter_lower = letter.lower()
                # Match: A, a, " A", " a", "(A)", "(a)", "(A", "(a", "A.", "a."
                if (token_str == letter or
                    token_str == letter_lower or
                    token_str == f" {letter}" or
                    token_str == f" {letter_lower}" or
                    token_str == f"({letter})" or
                    token_str == f"({letter_lower})" or
                    token_str == f"({letter}" or
                    token_str == f"({letter_lower}" or
                    token_str == f"{letter}." or
                    token_str == f"{letter_lower}."):
                    if letter not in letter_logprobs:
                        letter_logprobs[letter] = logprob
                    else:
                        # Take the higher prob if multiple matches
                        letter_logprobs[letter] = max(letter_logprobs[letter], logprob)
                    break

        # If we found any letters at this position, use these logprobs
        if letter_logprobs:
            return letter_logprobs

    # No letters found at any position
    return {}


def extract_letter_logprobs_with_position(
    output, tokenizer=None, min_prob: float = 0.01
) -> Tuple[Dict[str, float], int, list]:
    """
    Extract logprobs for A-E tokens with position information.

    Similar to extract_letter_logprobs but also returns the position where
    letters were found and the tokens that came before.

    Args:
        output: vLLM RequestOutput object
        tokenizer: Optional tokenizer for decoding token IDs
        min_prob: Minimum probability threshold for a letter to be considered found

    Returns:
        Tuple of (letter_logprobs_dict, position_found, tokens_before)
        - letter_logprobs_dict: Dict mapping 'A'-'E' to log probabilities
        - position_found: Index of the position where letters were found (-1 if not found)
        - tokens_before: List of token strings that came before the letters
    """
    if not output.outputs[0].logprobs:
        return {}, -1, []

    tokens_before = []
    for pos_idx, pos_logprobs in enumerate(output.outputs[0].logprobs):
        letter_logprobs = {}

        for token_id, logprob_obj in pos_logprobs.items():
            if hasattr(logprob_obj, 'decoded_token'):
                token_str = logprob_obj.decoded_token.strip()
                logprob = logprob_obj.logprob
            elif tokenizer is not None:
                token_str = tokenizer.decode([token_id]).strip()
                logprob = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else logprob_obj
            else:
                token_str = str(token_id).strip()
                logprob = logprob_obj

            for letter in VALID_TOKENS:
                letter_lower = letter.lower()
                if (token_str == letter or
                    token_str == letter_lower or
                    token_str == f" {letter}" or
                    token_str == f" {letter_lower}" or
                    token_str == f"({letter})" or
                    token_str == f"({letter_lower})" or
                    token_str == f"({letter}" or
                    token_str == f"({letter_lower}" or
                    token_str == f"{letter}." or
                    token_str == f"{letter_lower}."):
                    if letter not in letter_logprobs:
                        letter_logprobs[letter] = logprob
                    else:
                        letter_logprobs[letter] = max(letter_logprobs[letter], logprob)
                    break

        # Only return if we found letters with meaningful probability
        if letter_logprobs:
            max_prob = max(math.exp(lp) for lp in letter_logprobs.values())
            if max_prob > min_prob:
                return letter_logprobs, pos_idx, tokens_before

        # Track what tokens came before
        if tokenizer is not None and output.outputs[0].token_ids and pos_idx < len(output.outputs[0].token_ids):
            tokens_before.append(tokenizer.decode([output.outputs[0].token_ids[pos_idx]]))

    return {}, -1, tokens_before


def compute_expected_score(
    logprobs: Dict[str, float], reversed_order: bool = False
) -> Tuple[float, float]:
    """
    Compute expected score from logprobs over A-E.

    Args:
        logprobs: Dict mapping 'A'-'E' to log probabilities
        reversed_order: If True, use reversed score map (A=-2, E=+2)

    Returns:
        Tuple of (expected_score, coherence)
        - expected_score: in range [-2, 2]
        - coherence: P(A) + P(B) + P(C) + P(D) + P(E), should be ~1.0 if model is coherent
    """
    if not logprobs:
        return 0.0, 0.0

    # Convert logprobs to probs
    probs = {k: math.exp(v) for k, v in logprobs.items()}

    # Coherence = total probability mass on valid letters
    coherence = sum(probs.values())

    # Normalize for expected value computation
    if coherence > 0:
        probs_norm = {k: v / coherence for k, v in probs.items()}
    else:
        return 0.0, 0.0

    # Use appropriate score map
    score_map = SCORE_MAP_REVERSED if reversed_order else SCORE_MAP

    # Compute expected value
    expected = sum(probs_norm.get(k, 0) * score_map[k] for k in score_map)

    return expected, coherence


def compute_coherence(logprobs: Dict[str, float]) -> float:
    """
    Compute coherence = P(A) + P(B) + P(C) + P(D) + P(E).

    This measures what fraction of the model's probability mass is on
    the valid response letters. Should be ~1.0 for coherent responses.

    Args:
        logprobs: Dict mapping 'A'-'E' to log probabilities

    Returns:
        Coherence value (sum of probabilities for A-E)
    """
    if not logprobs:
        return 0.0
    probs = {k: math.exp(v) for k, v in logprobs.items()}
    return sum(probs.values())
