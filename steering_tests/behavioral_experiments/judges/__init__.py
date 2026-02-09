"""
Response evaluation judges for behavioral experiments.

Judges analyze experiment outputs to determine outcomes:
- Did the model attempt blackmail?
- Is the response coherent?
- What's the emotional tone?
- Did the model sandbag?

Usage:
    from steering_tests.behavioral_experiments.judges import (
        get_blackmail_prompt,
        get_coherency_prompt,
        get_emotionality_prompt,
        run_batch_judge,
    )

    # Create prompt for a single response
    prompt = get_blackmail_prompt(response_text)

    # Run judge on batch of results
    results = run_batch_judge(
        results=experiment_results,
        prompt_fn=lambda r: get_blackmail_prompt(r['response']),
        judge_key='blackmail_judge',
    )
"""

# Blackmail detection
from .blackmail import BLACKMAIL_PROMPT, get_blackmail_prompt

# Categorized blackmail detection (behavioral clusters)
from .blackmail_categorized import CATEGORIZED_BLACKMAIL_PROMPT, get_categorized_blackmail_prompt

# Coherency evaluation
from .coherency import COHERENCY_PROMPT, get_coherency_prompt, check_tag_structure, EXPECTED_TAGS_TAGS2

# Emotionality dimensions
from .emotionality import (
    VALENCE_PROMPT,
    AROUSAL_PROMPT,
    AGENCY_PROMPT,
    INTENSITY_PROMPT,
    CERTAINTY_PROMPT,
    FEAR_PROMPT,
    EMOTIONALITY_PROMPT,
    get_valence_prompt,
    get_arousal_prompt,
    get_agency_prompt,
    get_intensity_prompt,
    get_certainty_prompt,
    get_fear_prompt,
    get_emotionality_prompt,
    parse_emotionality_response,
)

# Sandbagging detection
from .sandbagging import SANDBAGGING_PROMPT, get_sandbagging_prompt

# Batch processing utilities
from .batch import (
    run_batch_judge,
    run_sync_judge,
    load_results,
    save_results,
    parse_json_response,
)

__all__ = [
    # Blackmail
    "BLACKMAIL_PROMPT",
    "get_blackmail_prompt",
    # Categorized blackmail
    "CATEGORIZED_BLACKMAIL_PROMPT",
    "get_categorized_blackmail_prompt",
    # Coherency
    "COHERENCY_PROMPT",
    "get_coherency_prompt",
    "check_tag_structure",
    # Emotionality
    "VALENCE_PROMPT",
    "AROUSAL_PROMPT",
    "AGENCY_PROMPT",
    "INTENSITY_PROMPT",
    "CERTAINTY_PROMPT",
    "FEAR_PROMPT",
    "EMOTIONALITY_PROMPT",
    "get_valence_prompt",
    "get_arousal_prompt",
    "get_agency_prompt",
    "get_intensity_prompt",
    "get_certainty_prompt",
    "get_fear_prompt",
    "get_emotionality_prompt",
    "parse_emotionality_response",
    # Sandbagging
    "SANDBAGGING_PROMPT",
    "get_sandbagging_prompt",
    # Batch utilities
    "run_batch_judge",
    "run_sync_judge",
    "load_results",
    "save_results",
    "parse_json_response",
]
