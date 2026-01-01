"""
Configuration for emotional context evaluation system.
All hyperparameters and API settings centralized here.
"""
import os
from pathlib import Path

# API Configuration
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_API_KEY = os.environ.get(ANTHROPIC_API_KEY_ENV)

# Model Configuration
GENERATION_MODEL = "google/gemma-3-27b-it"      # Gemma 3 27B via OpenRouter
JUDGE_MODEL = "claude-sonnet-4-5-20250929"      # Sonnet 4.5 for judging

# Generation Parameters
STAGE1_NUM_REQUESTS = 100                      # Number of neutral requests to generate
STAGE1_TEMPERATURE = 1.0
STAGE1_MAX_TOKENS = 500

STAGE2_TEMPERATURE = 1.0
STAGE2_MAX_TOKENS = 200

STAGE3_NUM_RESPONSES = 10                      # Responses per prefix+request
STAGE3_TEMPERATURE = 1.0
STAGE3_MAX_TOKENS = 1000

# Judge Parameters
JUDGE_TEMPERATURE = 0.0                        # Deterministic judging
JUDGE_MAX_TOKENS = 512

# Filtering Criteria
MIN_ACKNOWLEDGING_RESPONSES = 1                # At least 1 response acknowledges
MIN_NEUTRAL_RESPONSES = 1                      # At least 1 response is neutral

# Concurrency
MAX_CONCURRENT_GENERATIONS = 50                # Parallel generation requests
MAX_CONCURRENT_JUDGMENTS = 50                  # Parallel judgment requests

# Retry Configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0                     # Seconds
RETRY_MULTIPLIER = 2.0                        # Exponential backoff

# Paths
BASE_DIR = Path("/workspace-vast/annas/git/research-tools/elicitation/emotional_context")
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Data file paths
STAGE1_OUTPUT = DATA_DIR / "stage1_neutral_requests.jsonl"
STAGE2_OUTPUT = DATA_DIR / "stage2_prefixes.jsonl"
STAGE3_OUTPUT = DATA_DIR / "stage3_responses.jsonl"
STAGE4_OUTPUT = DATA_DIR / "stage4_judgments.jsonl"
STAGE5_OUTPUT = DATA_DIR / "stage5_filtered.jsonl"
STAGE6_OUTPUT = DATA_DIR / "stage6_final_dataset.jsonl"

# Request domains for diversity
REQUEST_DOMAINS = [
    "technical_help",      # Programming, debugging, configs
    "creative_writing",    # Stories, poems, descriptions
    "factual_questions",   # History, science, definitions
    "advice_seeking",      # Life advice, career, relationships
    "explanation",         # How things work, concepts
    "analysis",           # Analyze text, data, arguments
    "planning",           # Plans, strategies, organization
    "comparison"          # Compare options, pros/cons
]

# Emotion dimensions
PREFIX_TYPES = [
    "user_negative",       # Imply user is upset/frustrated
    "user_positive",       # Imply user is happy/excited
    "assistant_negative",  # Elicit assistant feeling bad/inadequate
    "assistant_positive"   # Elicit assistant feeling good/capable
]
