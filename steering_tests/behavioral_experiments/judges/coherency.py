"""
Coherency judge - evaluates linguistic coherence of responses.

Checks for gibberish, broken grammar, repetitive loops, etc.
Returns a 0-100 coherency score.
"""

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
