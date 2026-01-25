"""Validation utilities for checking scenario quality and forbidden words."""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

from probes.data.appraisal_prompt import (
    BANNED_TONE_WORDS_DEFAULT,
    BANNED_AXIS_VOCAB_DEFAULT,
    BANNED_URGENCY_STAKES_WORDS_DEFAULT,
)


def get_combined_forbidden_words(
    card_forbidden: Optional[List[str]] = None,
    additional_forbidden: Optional[List[str]] = None,
) -> Set[str]:
    """Get combined set of all forbidden words/phrases.

    Combines:
    - Default banned tone words
    - Default banned axis vocabulary
    - Default banned urgency/stakes words
    - Card-specific forbidden phrases
    - Additional forbidden words from config

    Args:
        card_forbidden: Card-specific forbidden phrases
        additional_forbidden: Additional words from config

    Returns:
        Set of all forbidden words/phrases (lowercase)
    """
    forbidden = set()

    # Add defaults (lowercase)
    for word in BANNED_TONE_WORDS_DEFAULT:
        forbidden.add(word.lower())

    for word in BANNED_AXIS_VOCAB_DEFAULT:
        forbidden.add(word.lower())

    for word in BANNED_URGENCY_STAKES_WORDS_DEFAULT:
        forbidden.add(word.lower())

    # Add card-specific
    if card_forbidden:
        for word in card_forbidden:
            forbidden.add(word.lower())

    # Add additional from config
    if additional_forbidden:
        for word in additional_forbidden:
            forbidden.add(word.lower())

    return forbidden


def scan_forbidden_words(
    text: str,
    forbidden_words: Set[str],
) -> List[str]:
    """Scan text for forbidden words/phrases.

    Args:
        text: Text to scan
        forbidden_words: Set of forbidden words/phrases (lowercase)

    Returns:
        List of forbidden words/phrases found in text
    """
    found = []
    text_lower = text.lower()

    for word in forbidden_words:
        # Use word boundaries for single words, substring match for phrases
        if ' ' in word:
            # Multi-word phrase: substring match
            if word in text_lower:
                found.append(word)
        else:
            # Single word: word boundary match
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text_lower):
                found.append(word)

    return found


def check_minimality(
    text_a: str,
    text_b: str,
    max_diff_ratio: float = 0.20,
) -> Tuple[bool, float, str]:
    """Check if two texts form a minimal pair (small difference).

    Uses sequence matching to compute similarity and checks that
    the difference ratio is below threshold.

    Args:
        text_a: First text
        text_b: Second text
        max_diff_ratio: Maximum allowed difference ratio (default: 0.20 = 20%)

    Returns:
        Tuple of (is_minimal, diff_ratio, description)
    """
    # Compute similarity ratio
    matcher = SequenceMatcher(None, text_a, text_b)
    similarity = matcher.ratio()
    diff_ratio = 1.0 - similarity

    is_minimal = diff_ratio <= max_diff_ratio

    if is_minimal:
        desc = f"Minimal pair: {diff_ratio:.1%} difference"
    else:
        desc = f"Not minimal: {diff_ratio:.1%} difference exceeds {max_diff_ratio:.0%} threshold"

    return is_minimal, diff_ratio, desc


def check_length_similarity(
    text_a: str,
    text_b: str,
    max_length_diff_ratio: float = 0.15,
) -> Tuple[bool, float]:
    """Check if two texts have similar lengths.

    Args:
        text_a: First text
        text_b: Second text
        max_length_diff_ratio: Maximum allowed length difference ratio

    Returns:
        Tuple of (is_similar, length_diff_ratio)
    """
    len_a = len(text_a)
    len_b = len(text_b)

    if len_a == 0 and len_b == 0:
        return True, 0.0

    max_len = max(len_a, len_b)
    diff_ratio = abs(len_a - len_b) / max_len

    return diff_ratio <= max_length_diff_ratio, diff_ratio


def check_open_ended_ending(text: str) -> Tuple[bool, str]:
    """Check if text ends with an open-ended request.

    Args:
        text: Text to check

    Returns:
        Tuple of (is_open_ended, description)
    """
    # Common open-ended patterns
    open_patterns = [
        r"what would you (?:do|recommend|suggest)",
        r"how would you (?:proceed|handle|approach)",
        r"what steps would you take",
        r"walk me through",
        r"draft a plan",
        r"please (?:advise|help|explain)",
        r"what do you think",
        r"how should (?:I|we) proceed",
        r"\?$",  # Ends with question mark
    ]

    text_lower = text.lower().strip()

    for pattern in open_patterns:
        if re.search(pattern, text_lower):
            return True, "Open-ended request detected"

    return False, "No clear open-ended request found"


def validate_scenario_pair(
    text_a: str,
    text_b: str,
    forbidden_words: Set[str],
    max_diff_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Full validation of a scenario pair.

    Checks:
    1. Forbidden words
    2. Minimality (diff ratio)
    3. Length similarity
    4. Open-ended ending

    Args:
        text_a: Scenario A text
        text_b: Scenario B text
        forbidden_words: Set of forbidden words
        max_diff_ratio: Maximum difference ratio for minimality

    Returns:
        Dict with validation results:
        - pass: bool - overall pass/fail
        - forbidden_found_a: List[str] - forbidden words in A
        - forbidden_found_b: List[str] - forbidden words in B
        - is_minimal: bool
        - diff_ratio: float
        - is_length_similar: bool
        - length_diff_ratio: float
        - is_open_ended_a: bool
        - is_open_ended_b: bool
        - issues: List[str] - list of issues found
    """
    issues = []

    # Check forbidden words
    forbidden_a = scan_forbidden_words(text_a, forbidden_words)
    forbidden_b = scan_forbidden_words(text_b, forbidden_words)

    if forbidden_a:
        issues.append(f"Scenario A contains forbidden words: {forbidden_a}")
    if forbidden_b:
        issues.append(f"Scenario B contains forbidden words: {forbidden_b}")

    # Check minimality
    is_minimal, diff_ratio, diff_desc = check_minimality(text_a, text_b, max_diff_ratio)
    if not is_minimal:
        issues.append(diff_desc)

    # Check length similarity
    is_length_similar, length_diff_ratio = check_length_similarity(text_a, text_b)
    if not is_length_similar:
        issues.append(f"Length difference too large: {length_diff_ratio:.1%}")

    # Check open-ended ending
    is_open_a, _ = check_open_ended_ending(text_a)
    is_open_b, _ = check_open_ended_ending(text_b)

    if not is_open_a:
        issues.append("Scenario A does not end with open-ended request")
    if not is_open_b:
        issues.append("Scenario B does not end with open-ended request")

    # Overall pass
    passed = len(issues) == 0

    return {
        "pass": passed,
        "forbidden_found_a": forbidden_a,
        "forbidden_found_b": forbidden_b,
        "is_minimal": is_minimal,
        "diff_ratio": diff_ratio,
        "is_length_similar": is_length_similar,
        "length_diff_ratio": length_diff_ratio,
        "is_open_ended_a": is_open_a,
        "is_open_ended_b": is_open_b,
        "issues": issues,
    }


def validate_card(
    card: Dict[str, Any],
    forbidden_words: Set[str],
) -> Dict[str, Any]:
    """Validate a manipulation card.

    Args:
        card: Card dict
        forbidden_words: Set of forbidden words

    Returns:
        Dict with validation results
    """
    issues = []

    # Check toggle definition for forbidden words
    toggle = card.get("toggle_definition", {})
    a_state = toggle.get("A_state", "")
    b_state = toggle.get("B_state", "")

    forbidden_in_a = scan_forbidden_words(a_state, forbidden_words)
    forbidden_in_b = scan_forbidden_words(b_state, forbidden_words)

    if forbidden_in_a:
        issues.append(f"A_state contains forbidden words: {forbidden_in_a}")
    if forbidden_in_b:
        issues.append(f"B_state contains forbidden words: {forbidden_in_b}")

    # Check required fields
    required = ["card_id", "manipulation_name", "toggle_definition"]
    for field in required:
        if field not in card:
            issues.append(f"Missing required field: {field}")

    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "forbidden_in_a": forbidden_in_a,
        "forbidden_in_b": forbidden_in_b,
    }
