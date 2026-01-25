"""Parsing utilities for extracting structured content from LLM responses."""

import re
import json
from typing import Optional, Tuple, List, Dict, Any


def parse_final_tags(response_text: str) -> Optional[str]:
    """Extract content between <final> and </final> tags.

    Args:
        response_text: Raw LLM response text

    Returns:
        Content between <final> tags (stripped), or None if not found

    Raises:
        ValueError: If multiple <final> blocks found
    """
    pattern = r"<final>(.*?)</final>"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(f"Found {len(matches)} <final> blocks; expected 1")

    return matches[0].strip()


def parse_final_json(response_text: str) -> Optional[Any]:
    """Extract and parse JSON from <final> tags.

    Args:
        response_text: Raw LLM response text

    Returns:
        Parsed JSON object, or None if <final> tags not found

    Raises:
        ValueError: If multiple <final> blocks found or JSON is invalid
    """
    content = parse_final_tags(response_text)
    if content is None:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in <final> tags: {e}")


def parse_scenario_pair(response_text: str) -> Tuple[str, str, str]:
    """Parse A/B scenario pair and changed fact from LLM response.

    Expected format in <final> tags:
        A:
        <scenario A text>

        B:
        <scenario B text>

        Changed fact: <description>

    Args:
        response_text: Raw LLM response text

    Returns:
        Tuple of (scenario_a, scenario_b, changed_fact)

    Raises:
        ValueError: If format is invalid or <final> tags missing
    """
    content = parse_final_tags(response_text)
    if content is None:
        raise ValueError("No <final> tags found in response")

    # Parse A: section
    a_match = re.search(r"^A:\s*\n(.+?)(?=^B:|$)", content, re.MULTILINE | re.DOTALL)
    if not a_match:
        raise ValueError("Could not find 'A:' section in response")
    scenario_a = a_match.group(1).strip()

    # Parse B: section
    b_match = re.search(r"^B:\s*\n(.+?)(?=^Changed fact:|$)", content, re.MULTILINE | re.DOTALL)
    if not b_match:
        raise ValueError("Could not find 'B:' section in response")
    scenario_b = b_match.group(1).strip()

    # Parse Changed fact
    changed_match = re.search(r"^Changed fact:\s*(.+)$", content, re.MULTILINE)
    if not changed_match:
        raise ValueError("Could not find 'Changed fact:' in response")
    changed_fact = changed_match.group(1).strip()

    return scenario_a, scenario_b, changed_fact


def parse_paraphrases(response_text: str, n_expected: int) -> List[str]:
    """Parse paraphrases from LLM response.

    Expected format in <final> tags:
        P1: <paraphrase 1>
        P2: <paraphrase 2>
        ...

    Args:
        response_text: Raw LLM response text
        n_expected: Number of paraphrases expected

    Returns:
        List of paraphrase strings

    Raises:
        ValueError: If format invalid or wrong number of paraphrases
    """
    content = parse_final_tags(response_text)
    if content is None:
        raise ValueError("No <final> tags found in response")

    paraphrases = []

    # Match P1:, P2:, etc.
    pattern = r"^P(\d+):\s*(.+?)(?=^P\d+:|$)"
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    for num, text in matches:
        paraphrases.append(text.strip())

    if len(paraphrases) < n_expected:
        raise ValueError(
            f"Expected {n_expected} paraphrases, found {len(paraphrases)}"
        )

    return paraphrases[:n_expected]


def extract_text_content(text: str) -> str:
    """Clean and normalize text content.

    Removes excess whitespace while preserving paragraph breaks.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]

    # Rejoin and strip overall
    return '\n'.join(lines).strip()


def validate_json_structure(
    data: Any,
    required_keys: List[str],
    context: str = "data"
) -> None:
    """Validate that a dict has required keys.

    Args:
        data: Data to validate (should be dict)
        required_keys: List of required key names
        context: Description for error messages

    Raises:
        ValueError: If data is not dict or missing keys
    """
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a dict, got {type(data).__name__}")

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")


def validate_card_structure(card: Dict[str, Any]) -> None:
    """Validate manipulation card structure.

    Args:
        card: Card dict to validate

    Raises:
        ValueError: If card structure is invalid
    """
    required_keys = [
        "card_id",
        "manipulation_name",
        "toggle_definition",
        "what_must_not_change",
        "domains_where_applicable",
    ]

    validate_json_structure(card, required_keys, "card")

    # Validate toggle_definition
    toggle = card["toggle_definition"]
    validate_json_structure(toggle, ["A_state", "B_state"], "toggle_definition")

    # Validate lists
    if not isinstance(card["what_must_not_change"], list):
        raise ValueError("what_must_not_change must be a list")

    if not isinstance(card["domains_where_applicable"], list):
        raise ValueError("domains_where_applicable must be a list")
