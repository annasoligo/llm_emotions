"""
Logit-based emotion detection using logit lens methodology.

This module provides a clean, refactored implementation of emotion detection
by projecting hidden states to vocabulary logits and measuring emotion word scores.

Key Features:
- Project hidden states to vocabulary logits
- Extract logits for emotion-related tokens
- Z-score normalization using baseline statistics
- Mean/max aggregation methods
- Optimized batched operations

Example usage:
    >>> from emotion_logit_lens import (
    ...     EmotionTokenManager,
    ...     LogitBaselineLoader,
    ...     project_to_logits,
    ...     compute_emotion_scores_from_logits
    ... )
    >>>
    >>> # Load emotion token IDs
    >>> emotion_mgr = EmotionTokenManager(model_name="google_gemma_3_27b_it")
    >>> emotion_token_ids = emotion_mgr.load_emotion_token_ids()
    >>>
    >>> # Load baseline statistics
    >>> baseline_loader = LogitBaselineLoader(
    ...     baseline_dir="data/baselines/logit_emotion_alpaca",
    ...     model_name="google_gemma_3_27b_it"
    ... )
    >>>
    >>> # Project activation to logits
    >>> logits = project_to_logits(model, activation_vector)
    >>>
    >>> # Compute emotion scores
    >>> baseline_stats = {'layers_data': {
    ...     '30': {'statistics': baseline_loader.load_layer_stats(30)}
    ... }}
    >>> scores = compute_emotion_scores_from_logits(
    ...     logits=logits,
    ...     emotion_token_ids=emotion_token_ids,
    ...     baseline_stats=baseline_stats,
    ...     layer=30,
    ...     aggregation='mean'
    ... )
    >>> print(scores)
    {'anger': 0.34, 'happiness': 1.82, 'sadness': -0.45, ...}
"""

from .config import (
    DEFAULT_MODEL,
    EKMAN6_EMOTIONS,
    get_research_tools_root,
    get_baseline_dir,
    get_emotion_token_ids_dir,
)

from .emotion_tokens import EmotionTokenManager

from .baseline_loader import LogitBaselineLoader

from .core import (
    project_to_logits,
    project_to_logits_batched,
    compute_emotion_scores_from_logits,
    compute_emotion_scores_batched,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    'DEFAULT_MODEL',
    'EKMAN6_EMOTIONS',
    'get_research_tools_root',
    'get_baseline_dir',
    'get_emotion_token_ids_dir',
    # Emotion tokens
    'EmotionTokenManager',
    # Baseline loading
    'LogitBaselineLoader',
    # Core functions
    'project_to_logits',
    'project_to_logits_batched',
    'compute_emotion_scores_from_logits',
    'compute_emotion_scores_batched',
]
