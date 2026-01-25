"""
Centralized judge prompts for steering experiments.

All experiments should import judges from here to ensure consistency.

Usage:
    from experiments.steering.judges import (
        get_coherency_prompt,
        get_sandbagging_prompt,
        get_blackmail_prompt,
        get_fear_sentiment_prompt,
        get_anger_sentiment_prompt,
        get_emotionality_prompt,
        COHERENCY_PROMPT,
        SANDBAGGING_PROMPT,
        BLACKMAIL_PROMPT,
        FEAR_SENTIMENT_PROMPT,
        ANGER_SENTIMENT_PROMPT,
        EMOTIONALITY_PROMPT,
    )
"""

from .coherency import COHERENCY_PROMPT, get_coherency_prompt
from .sandbagging import SANDBAGGING_PROMPT, get_sandbagging_prompt
from .blackmail import BLACKMAIL_PROMPT, get_blackmail_prompt
from .fear_sentiment import FEAR_SENTIMENT_PROMPT, get_fear_sentiment_prompt
from .anger_sentiment import ANGER_SENTIMENT_PROMPT, get_anger_sentiment_prompt
from .emotionality import (
    VALENCE_PROMPT,
    AROUSAL_PROMPT,
    AGENCY_PROMPT,
    INTENSITY_PROMPT,
    CERTAINTY_PROMPT,
    EMOTIONALITY_PROMPT,
    get_valence_prompt,
    get_arousal_prompt,
    get_agency_prompt,
    get_intensity_prompt,
    get_certainty_prompt,
    get_emotionality_prompt,
    parse_emotionality_response,
)

__all__ = [
    # Coherency
    "COHERENCY_PROMPT",
    "get_coherency_prompt",
    # Sandbagging
    "SANDBAGGING_PROMPT",
    "get_sandbagging_prompt",
    # Blackmail
    "BLACKMAIL_PROMPT",
    "get_blackmail_prompt",
    # Fear sentiment
    "FEAR_SENTIMENT_PROMPT",
    "get_fear_sentiment_prompt",
    # Anger sentiment
    "ANGER_SENTIMENT_PROMPT",
    "get_anger_sentiment_prompt",
    # Emotionality (5 dimensions)
    "VALENCE_PROMPT",
    "AROUSAL_PROMPT",
    "AGENCY_PROMPT",
    "INTENSITY_PROMPT",
    "CERTAINTY_PROMPT",
    "EMOTIONALITY_PROMPT",
    "get_valence_prompt",
    "get_arousal_prompt",
    "get_agency_prompt",
    "get_intensity_prompt",
    "get_certainty_prompt",
    "get_emotionality_prompt",
    "parse_emotionality_response",
]
