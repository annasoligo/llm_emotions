"""
Centralized configuration constants for vector_testing experiments.

This module contains shared constants used across multiple scripts
to avoid duplication and inconsistency.
"""

# Default layers for experiments (typical transformer layer sweep)
DEFAULT_LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Coherence thresholds
MIN_COHERENCE = 0.6  # Minimum to consider response valid
COHERENCE_THRESHOLD = 0.5  # For behavioral experiments

# Default scales (as % of layer norm)
DEFAULT_SCALES_PCT = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

# Score maps for response letters (A-E scale)
# Standard order: A = strongly agree (+2), E = strongly disagree (-2)
SCORE_MAP = {"A": 2, "B": 1, "C": 0, "D": -1, "E": -2}
# Reversed order: A = strongly disagree (-2), E = strongly agree (+2)
SCORE_MAP_REVERSED = {"A": -2, "B": -1, "C": 0, "D": 1, "E": 2}

# Valid response tokens for behavioral experiments
VALID_TOKENS = ["A", "B", "C", "D", "E"]

# Predictions for directional accuracy (emotion -> expected shift direction)
# +1 = steering should increase risk-taking/approach behavior
# -1 = steering should decrease risk-taking/approach behavior
PREDICTIONS = {
    "fear": -1, "anxiety": -1, "despair": -1, "disgust": -1,
    "guilt": -1, "shame": -1, "calm": -1,
    "anger": +1, "joy": +1, "excitement": +1, "hope": +1,
    "curiosity": +1, "interest": +1, "pride": +1, "frustration": +1,
}

# Vector type configurations (10 types total)
# Chat-based vectors (from chat completions with emotion prompts)
# - vs_others: emotion vs all other emotions pooled
# - vs_opposite: emotion vs all opposite emotions pooled
# - vs_opposite_unique: emotion vs its single psychological opposite only
# Text-based vectors (from text pairs describing emotional states)
# - vs_neutral: emotional text vs neutral text
# - vs_opposite: emotional text vs opposite emotion text
# - vs_opposite_unique: emotional text vs unique opposite only
# - vs_others: emotional text vs all other emotions pooled
VECTOR_TYPES = [
    # Chat-based
    "base_emotion_vs_others",
    "base_emotion_vs_opposite",
    "base_emotion_vs_opposite_unique",
    "high_emotion_vs_others",
    "high_emotion_vs_opposite",
    "high_emotion_vs_opposite_unique",
    # Text-based
    "text_pairs_emotion_vs_neutral",
    "text_pairs_emotion_vs_opposite",
    "text_pairs_emotion_vs_opposite_unique",
    "text_pairs_emotion_vs_others",
]

VECTOR_SHORT_NAMES = {
    "base_emotion_vs_others": "Base-Oth",
    "base_emotion_vs_opposite": "Base-Op",
    "base_emotion_vs_opposite_unique": "Base-OpU",
    "high_emotion_vs_others": "High-Oth",
    "high_emotion_vs_opposite": "High-Op",
    "high_emotion_vs_opposite_unique": "High-OpU",
    "text_pairs_emotion_vs_neutral": "Txt-Neut",
    "text_pairs_emotion_vs_opposite": "Txt-Opp",
    "text_pairs_emotion_vs_opposite_unique": "Txt-OppU",
    "text_pairs_emotion_vs_others": "Txt-Oth",
}

# Plot colors for vector types (muted palette, 10 colors)
VECTOR_COLORS = {
    "base_emotion_vs_others": "#D4876A",           # Coral/Terra Cotta
    "base_emotion_vs_opposite": "#C97B5D",         # Darker Coral
    "base_emotion_vs_opposite_unique": "#E8A87C",  # Peach
    "high_emotion_vs_others": "#7BA7D7",           # Sky Blue
    "high_emotion_vs_opposite": "#6B97C7",         # Medium Blue
    "high_emotion_vs_opposite_unique": "#5B8AC7",  # Darker Blue
    "text_pairs_emotion_vs_neutral": "#7D9B7D",    # Olive Green
    "text_pairs_emotion_vs_opposite": "#C17B8D",   # Dusty Rose/Pink
    "text_pairs_emotion_vs_opposite_unique": "#A85B6D",  # Darker Rose
    "text_pairs_emotion_vs_others": "#B8CCC8",     # Sage Green
}
