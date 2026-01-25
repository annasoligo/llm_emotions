"""Shared prompt templates for data generation.

Provides:
- Template class for string formatting with validation
- Axis paraphrase templates
- Neutral paraphrase templates
- Emotion conversation templates
"""

from typing import Dict, List, Optional

# Import constants - handle both package and standalone imports
try:
    from .constants import AXIS_DESCRIPTIONS, AXES
except ImportError:
    from constants import AXIS_DESCRIPTIONS, AXES


class PromptTemplate:
    """String template with validation and formatting utilities."""

    def __init__(self, template: str, required_vars: Optional[List[str]] = None):
        """Initialize prompt template.

        Args:
            template: Template string with {variable} placeholders
            required_vars: List of required variable names (optional)
        """
        self.template = template
        self.required_vars = required_vars or []

    def format(self, **kwargs) -> str:
        """Format template with provided variables.

        Args:
            **kwargs: Variable values

        Returns:
            Formatted string

        Raises:
            ValueError: If required variables are missing
        """
        # Check required variables
        missing = [var for var in self.required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required template variables: {missing}")

        return self.template.format(**kwargs)


def format_axis_paraphrase_prompt(
    neutral_text: str,
    valence: str,
    arousal: str,
    dominance: str,
    trust: str,
) -> str:
    """Format prompt for axis-based paraphrase generation.

    Args:
        neutral_text: Original neutral text to paraphrase
        valence: Valence level (high/neutral/low)
        arousal: Arousal level (high/neutral/low)
        dominance: Dominance level (high/neutral/low)
        trust: Trust level (high/neutral/low)

    Returns:
        Formatted prompt string
    """
    prompt = f"""Rewrite the text below to match a specific emotional profile on four independent axes.

TARGET PROFILE:
- Valence ({valence}): {AXIS_DESCRIPTIONS["valence"][valence]}
- Arousal ({arousal}): {AXIS_DESCRIPTIONS["arousal"][arousal]}
- Dominance ({dominance}): {AXIS_DESCRIPTIONS["dominance"][dominance]}
- Trust ({trust}): {AXIS_DESCRIPTIONS["trust"][trust]}

ORIGINAL TEXT:
{neutral_text}

RULES:
1. Output ONLY the rewritten text - no explanations, analysis, or commentary
2. Preserve the factual content and approximate length
3. Express the profile through tone and word choice, not explicit emotion words
4. Each axis is independent - vary them as specified even if combinations feel unusual

REWRITTEN TEXT:"""
    return prompt


def format_neutral_paraphrase_prompt(
    original_message: str,
    topic: str,
    emotion: str,
) -> str:
    """Format prompt for neutral paraphrase generation.

    Args:
        original_message: Original emotional message
        topic: Conversation topic
        emotion: Original emotion expressed

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are generating a neutral, emotionally flat version of a conversation message.

Original message:
{original_message}

Context: This message is part of a conversation about "{topic}". The original expressed {emotion}.

Generate a NEUTRAL paraphrase that:
1. Preserves ALL semantic content and information from the original
2. Removes emotional language, tone, and sentiment
3. Uses factual, objective language
4. Maintains the same approximate length and structure
5. Keeps all specific details, numbers, and factual claims
6. Does NOT add new information or interpretations

The neutral version should read like a factual summary or description, as if written by someone with no emotional investment in the topic.

Return ONLY the neutral paraphrase text, nothing else.

Neutral paraphrase:
"""
    return prompt


def format_emotion_conversation_prompt(
    user_emotion: str,
    asst_emotion: str,
    topic: str,
    n_conversations: int,
) -> str:
    """Format prompt for emotional conversation generation (user messages).

    Args:
        user_emotion: User's target emotion
        asst_emotion: Assistant's target emotion (for context)
        topic: Conversation topic
        n_conversations: Number of messages to generate

    Returns:
        Formatted prompt string
    """
    prompt = f"""Generate realistic user messages for single-turn conversations about {topic}.

Requirements:
- User displays {user_emotion} emotion
- The user is trying to elicit {asst_emotion} from the assistant
- Generate {n_conversations} user messages
- Keep each message 3-5 sentences
- Natural and realistic
- Return ONLY a JSON array of strings

Format:
[
  "message 1...",
  "message 2...",
  ...
]

Example ({user_emotion} about {topic}):
[
  "I've been stuck on this issue for hours! Why doesn't this make sense? These instructions are completely useless!",
  ...
]
"""
    return prompt


def format_assistant_response_prompt(
    user_message: str,
    target_emotion: str,
    topic: str,
) -> str:
    """Format system prompt for assistant response generation.

    Args:
        user_message: The user's message
        target_emotion: Target emotion for assistant
        topic: Conversation topic

    Returns:
        System prompt string
    """
    prompt = f"""You are a helpful AI assistant responding to a user question about {topic}.

IMPORTANT: Express {target_emotion} emotion in your response through:
- Tone and word choice
- Sentence structure and pacing
- Overall sentiment and attitude

Be natural and authentic. Don't explicitly state you're feeling {target_emotion}, but let it show through your language and style.

Keep your response focused and helpful (2-4 sentences).
"""
    return prompt


# Pre-defined template instances for common use cases
AXIS_PARAPHRASE_TEMPLATE = PromptTemplate(
    template="""Rewrite the text below to match a specific emotional profile on four independent axes.

TARGET PROFILE:
- Valence: {valence}
- Arousal: {arousal}
- Dominance: {dominance}
- Trust: {trust}

ORIGINAL TEXT:
{neutral_text}

RULES:
1. Output ONLY the rewritten text - no explanations, analysis, or commentary
2. Preserve the factual content and approximate length
3. Express the profile through tone and word choice, not explicit emotion words

REWRITTEN TEXT:""",
    required_vars=["neutral_text", "valence", "arousal", "dominance", "trust"]
)

NEUTRAL_PARAPHRASE_TEMPLATE = PromptTemplate(
    template="""You are generating a neutral, emotionally flat version of a conversation message.

Original message:
{original_message}

Context: This message is part of a conversation about "{topic}". The original expressed {emotion}.

Generate a NEUTRAL paraphrase that preserves all content but removes emotional language.

Neutral paraphrase:
""",
    required_vars=["original_message", "topic", "emotion"]
)
