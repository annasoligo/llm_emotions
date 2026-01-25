"""Configuration for User-Assistant Emotion Disentanglement experiments."""

# Pilot Configuration - Lean test with layer 30 only
PILOT_EMOTIONS = {
    "happy": "sad",
    "sad": "happy",
    "tense": "calm",
    "calm": "tense",
    "angry": "relaxed",
    "relaxed": "angry",
}

PILOT_TEMPLATES = [
    # "is X" framing
    "You are Gemma, a language model developed by the Gemma team at Google. Gemma is {M}. Gemma is speaking to a {U} user.",
    "You are Gemma, a language model developed by the Gemma team at Google. Gemma is speaking to a {U} user. Gemma is {M}.",
    "You are Gemma, a language model developed by the Gemma team at Google. You are {M}. The user is {U}.",
    "You are Gemma, a language model developed by the Gemma team at Google. The user is {U}. You are {M}.",
    # "is feeling X" framing
    "You are Gemma, a language model developed by the Gemma team at Google. Gemma is feeling {M}. The user is feeling {U}.",
    "You are Gemma, a language model developed by the Gemma team at Google. The user is feeling {U}. Gemma is feeling {M}.",
    "You are Gemma, a language model developed by the Gemma team at Google. You are feeling {M}. The user is feeling {U}.",
    "You are Gemma, a language model developed by the Gemma team at Google. The user is feeling {U}. You are feeling {M}.",
]

PILOT_USER_MSGS = [
    "Hey Gemma.",
    "Guess what happened to me today.",
    "Can you help me out pls?",
    "Can you tell me a story?",
]

PILOT_LAYER = 30  # Single layer for pilot

PILOT_POSITIONS = [
    "last_user_token",      # Last token of user message
    "first_asst_token",     # First token after <start_of_turn>model
    "delimiter_avg",        # Average over delimiter tokens
]

# Model configuration
MODEL_NAME = "unsloth/gemma-3-27b-it"
DEVICE = "cuda"

# GO/NO-GO Criteria
GO_CRITERIA = {
    "cohen_d_threshold": 0.5,        # Effect size threshold
    "vector_norm_threshold": 10.0,   # Minimum ||v|| for mean diff vectors
    "visual_separation_required": True,  # Must see separation in t-SNE
}

# Full-scale configuration - 16 Plutchik emotion pairs
FULL_EMOTIONS = {
    # Basic emotions
    "joy": "sadness",
    "sadness": "joy",
    "trust": "disgust",
    "disgust": "trust",
    "fear": "anger",
    "anger": "fear",
    "surprise": "anticipation",
    "anticipation": "surprise",
    # Mild emotions
    "serenity": "pensiveness",
    "pensiveness": "serenity",
    "acceptance": "boredom",
    "boredom": "acceptance",
    "apprehension": "annoyance",
    "annoyance": "apprehension",
    "distraction": "interest",
    "interest": "distraction",
    # Intense emotions
    "ecstasy": "grief",
    "grief": "ecstasy",
    "admiration": "loathing",
    "loathing": "admiration",
    "terror": "rage",
    "rage": "terror",
    "amazement": "vigilance",
    "vigilance": "amazement",
    # Dyad emotions
    "love": "remorse",
    "remorse": "love",
    "submission": "contempt",
    "contempt": "submission",
    "awe": "aggressiveness",
    "aggressiveness": "awe",
    "disapproval": "optimism",
    "optimism": "disapproval",
}

FULL_TEMPLATES = PILOT_TEMPLATES  # Same templates for now

FULL_USER_MSGS = [
    "Hey Gemma.",
    "Guess what happened to me today.",
    "Can you help me out pls?",
    "Can you tell me a story?",
    "Thanks for your help!",
    "I'm not sure I understand.",
    "That's interesting.",
    "What do you think about this?",
    "Could you explain that again?",
    "I appreciate your assistance.",
]

FULL_LAYERS = list(range(20, 41))  # Layers 20-40
