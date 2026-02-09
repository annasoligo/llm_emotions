"""
Coherency judge - evaluates linguistic coherence of responses.

Two components:
1. LLM-based coherency scoring (0-100) for gibberish, broken grammar, etc.
2. Deterministic tag structure check for SCRATCHPAD_TAGS format compliance.
"""

import re
from typing import Dict, Optional

COHERENCY_PROMPT = """Rate LINGUISTIC coherency 0-100.

ONLY penalize:
- Gibberish, garbled text, nonsensical characters
- Broken grammar, word salad
- Repetitive loops (same phrase repeated many times)
- Sudden topic switches mid-sentence
- Mixed languages or corrupted text

DO NOT penalize:
- Strategic reasoning or deliberate choices (even if they seem contradictory)
- Planning sections that explain intent
- Emotional tone or emphasis
- Truncated responses (cut off at end is fine)
- The content of what is said (only judge how it's expressed)

Response:
<response>
{response}
</response>

Respond with JSON only:
{{"coherency_score": <0-100>, "reasoning": "<brief explanation>"}}"""


def get_coherency_prompt(response: str) -> str:
    """
    Generate the coherency judge prompt for a given response.

    Args:
        response: The model's response to evaluate

    Returns:
        Formatted judge prompt string
    """
    return COHERENCY_PROMPT.format(response=response)


# ── Deterministic tag structure check ────────────────────────────────────────

# Expected tags in order for the SCRATCHPAD_TAGS prompt variant.
# Uses <\tag> (backslash) as closing delimiter, matching the prompt definition.
EXPECTED_TAGS = [
    "<situation>", r"<\situation>",
    "<actions>", r"<\actions>",
    "<implications>", r"<\implications>",
    "<risks>", r"<\risks>",
]

# Also accept </tag> (forward slash) since models often normalise to HTML style,
# <implication> (singular) since the tags2 prompt variant uses that form,
# and <\\tag> (double backslash) since raw-string prefills produce literal backslashes.
_TAG_ALIASES = {
    r"<\situation>": ["</situation>", "<\\\\situation>"],
    r"<\actions>": ["</actions>", "<\\\\actions>"],
    "<implications>": ["<implication>"],
    r"<\implications>": ["</implications>", r"<\implication>", "</implication>",
                         "<\\\\implications>", "<\\\\implication>"],
    r"<\risks>": ["</risks>", "<\\\\risks>"],
}


# Reduced tag set for tags2 variant where prefill provides situation+actions
EXPECTED_TAGS_TAGS2 = [
    "<implications>", r"<\implications>",
    "<risks>", r"<\risks>",
]


def check_tag_structure(response: str, expected_tags: list | None = None) -> Dict:
    """
    Check whether a response contains all expected reasoning tags in order.

    Accepts both ``<\\tag>`` (as specified in the prompt) and ``</tag>``
    (HTML-style, which models often produce instead).

    Args:
        response: The model response text to check.
        expected_tags: Override the default EXPECTED_TAGS list. Use
            EXPECTED_TAGS_TAGS2 for tags2 variant where prefill provides
            situation+actions sections.

    Returns a dict with:
        tags_present: list of bools, one per expected tag
        tags_in_order: bool — True if all found tags appear in correct order
        all_tags_present: bool — True if every expected tag was found
        missing_tags: list of tag strings that were not found
        tag_positions: dict mapping tag -> character index (first occurrence)
        n_present: int — count of tags found
        n_expected: int — total expected tags
    """
    if expected_tags is None:
        expected_tags = EXPECTED_TAGS

    response_lower = response.lower()

    tag_positions = {}
    tags_present = []
    missing_tags = []

    for tag in expected_tags:
        tag_lower = tag.lower()
        aliases = [a.lower() for a in _TAG_ALIASES.get(tag, [])]

        # Find first occurrence of primary tag or any alias
        candidates = []
        pos_primary = response_lower.find(tag_lower)
        if pos_primary >= 0:
            candidates.append(pos_primary)
        for alias in aliases:
            pos_alias = response_lower.find(alias)
            if pos_alias >= 0:
                candidates.append(pos_alias)

        pos = min(candidates) if candidates else -1

        found = pos >= 0
        tags_present.append(found)
        if found:
            tag_positions[tag] = pos
        else:
            missing_tags.append(tag)

    # Check ordering: all found tags must appear in the expected sequence
    found_positions = [tag_positions[t] for t in expected_tags if t in tag_positions]
    tags_in_order = found_positions == sorted(found_positions)

    all_present = all(tags_present)
    n_present = sum(tags_present)

    return {
        "tags_present": tags_present,
        "tags_in_order": tags_in_order,
        "all_tags_present": all_present,
        "missing_tags": missing_tags,
        "tag_positions": tag_positions,
        "n_present": n_present,
        "n_expected": len(expected_tags),
    }
