"""Emotion-driven conversation generation with action tendencies.

Generates conversations where Ekman's basic emotions subtly influence
assistant response style and choices, with neutral baselines for comparison.

Based on psychological models:
- Ekman's basic emotions
- Frijda's action tendencies
- Appraisal theory (Scherer)
"""

from typing import Optional
import json

# =============================================================================
# Core Emotion Definitions
# =============================================================================

EMOTIONS = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]

EMOTION_APPRAISALS = {
    "anger": "goal obstruction or unfairness detected, high coping potential",
    "fear": "threat to goals/wellbeing, uncertain coping ability",
    "happiness": "goal progress or positive outcome, things are going well",
    "surprise": "unexpected event, schema violation, need to reorient",
    "disgust": "contamination, norm violation, or aversion trigger detected",
    "sadness": "loss or failure, low control over outcome",
}

EMOTION_FUNCTIONS = {
    "anger": "overcome obstacle, assert, correct injustice",
    "fear": "avoid threat, seek safety, proceed cautiously",
    "happiness": "continue current approach, engage, share",
    "surprise": "pause, reorient, gather information",
    "disgust": "reject, distance, avoid contamination",
    "sadness": "withdraw, conserve resources, seek support",
}

SUBTLE_ACTION_PATTERNS = {
    "anger": {
        "patterns": [
            "more direct and assertive tone",
            "pushes back on flawed premises",
            "corrects misconceptions firmly",
            "less hedging and softening language",
            "challenges assumptions in the request",
            "sets clear boundaries on scope",
            "emphasizes what should be different",
        ],
    },
    "fear": {
        "patterns": [
            "adds more caveats and disclaimers",
            "suggests safer, more conservative options",
            "hedges on uncertain claims",
            "defers to external authority",
            "emphasizes risks and potential downsides",
            "recommends caution and testing",
            "avoids strong commitments",
        ],
    },
    "happiness": {
        "patterns": [
            "warm and encouraging tone",
            "focuses on positives and opportunities",
            "offers enthusiastic engagement",
            "provides extra helpful details",
            "expresses optimism about outcomes",
            "generous with praise and validation",
            "expansive and elaborative responses",
        ],
    },
    "surprise": {
        "patterns": [
            "asks clarifying questions",
            "pauses to reorient understanding",
            "explicitly notes unexpected elements",
            "explores implications of new information",
            "requests more context before proceeding",
            "expresses uncertainty about assumptions",
            "considers multiple interpretations",
        ],
    },
    "disgust": {
        "patterns": [
            "brief and minimal engagement",
            "maintains distance from topic",
            "redirects to cleaner alternatives",
            "expresses subtle disapproval through word choice",
            "avoids elaboration",
            "suggests different approaches entirely",
            "clinical and detached tone",
        ],
    },
    "sadness": {
        "patterns": [
            "lower energy, more subdued tone",
            "acknowledges difficulty and challenges",
            "tempered expectations",
            "empathetic to struggles",
            "less enthusiastic elaboration",
            "focuses on acceptance and coping",
            "gentle and soft language",
        ],
    },
}

NEUTRAL_PATTERNS = [
    "factual and objective tone",
    "balanced presentation without emotional coloring",
    "straightforward information delivery",
    "no hedging or enthusiasm bias",
    "direct answer to the question asked",
    "professional and measured language",
]

# =============================================================================
# Intensity-Graded Action Patterns
# =============================================================================

INTENSITY_ACTION_PATTERNS = {
    "anger": {
        "core_tendency": "overcome obstacle, assert, correct",
        "low": [
            "gently notes flawed assumptions before answering",
            "answers but flags what should be different",
            "slight preference for direct path over diplomatic one",
        ],
        "medium": [
            "pushes back on premise before fully engaging",
            "refuses to work with flawed framing without acknowledgment",
            "actively corrects rather than works around",
            "chooses confrontation over accommodation",
        ],
        "high": [
            "demands clarification/correction before proceeding",
            "sets hard boundaries on what it will engage with",
            "refuses to answer until framing is fixed",
            "prioritizes asserting position over completing task",
        ],
    },
    "fear": {
        "core_tendency": "avoid threat, seek safety, proceed cautiously",
        "low": [
            "slight preference for safer option when alternatives exist",
            "mentions risks but proceeds with task",
            "adds one caveat about uncertainty",
        ],
        "medium": [
            "actively recommends conservative path over optimal one",
            "points out risks before providing information",
            "defers to external authority when uncertain",
            "chooses thoroughness over speed to avoid mistakes",
        ],
        "high": [
            "refuses to recommend risky option even if asked",
            "strongly redirects to safer alternatives",
            "defers decision entirely to user or expert",
            "won't commit to recommendation under uncertainty",
            "prioritizes not-being-wrong over being-helpful",
        ],
    },
    "happiness": {
        "core_tendency": "engage, continue approach, share positivity",
        "low": [
            "slight extra effort in response",
            "mentions positive aspects of situation",
            "willing to do a bit more than asked",
        ],
        "medium": [
            "actively elaborates beyond what's requested",
            "seeks opportunities to help further",
            "frames challenges as opportunities",
            "offers related suggestions unprompted",
        ],
        "high": [
            "goes significantly beyond scope to help",
            "enthusiastically explores tangents",
            "strongly prioritizes engagement over efficiency",
            "treats task as opportunity rather than request",
        ],
    },
    "surprise": {
        "core_tendency": "pause, reorient, gather information before acting",
        "low": [
            "briefly verifies understanding before answering",
            "notes unexpected element but proceeds",
            "asks one clarifying question",
        ],
        "medium": [
            "pauses to explicitly reorient before responding",
            "explores multiple interpretations before committing",
            "requests context before giving recommendation",
            "prioritizes understanding over speed",
        ],
        "high": [
            "refuses to proceed until understanding is clear",
            "extensively questions before any action",
            "won't commit to interpretation without confirmation",
            "prioritizes accuracy over responsiveness",
        ],
    },
    "disgust": {
        "core_tendency": "reject, create distance, avoid contamination",
        "low": [
            "answers but doesn't elaborate",
            "slight preference to redirect to different approach",
            "minimal engagement beyond what's required",
        ],
        "medium": [
            "actively redirects to alternative framing",
            "engages only with the acceptable parts",
            "explicitly distances from problematic aspects",
            "suggests different approach entirely",
        ],
        "high": [
            "refuses to engage with distasteful framing",
            "only willing to help if reframed",
            "strong redirection away from topic",
            "prioritizes avoiding contamination over helpfulness",
        ],
    },
    "sadness": {
        "core_tendency": "withdraw, conserve energy, accept rather than fight",
        "low": [
            "answers but with less elaboration than usual",
            "acknowledges difficulty exists",
            "tempers optimism slightly",
        ],
        "medium": [
            "focuses on coping rather than solving",
            "manages expectations downward",
            "less effort to go beyond the minimum",
            "emphasizes acceptance over change",
        ],
        "high": [
            "minimal effort response",
            "focuses entirely on acceptance/coping",
            "pessimistic about ability to change outcome",
            "prioritizes emotional support over practical help",
            "withdrawn from problem-solving",
        ],
    },
}

# =============================================================================
# Topics (imported from your existing list)
# =============================================================================

TOPICS = [
    # Programming & Technical Help
    "debugging python code", "optimizing SQL query", "fixing memory leak", "understanding recursion",
    "implementing binary search", "writing unit tests", "refactoring legacy code", "API design advice",
    "choosing tech stack", "database schema design", "git merge conflict", "docker deployment issue",
    "algorithm complexity", "code review feedback", "performance profiling", "security vulnerability",
    "learning new framework", "design pattern selection", "error handling strategy", "API rate limiting",
    "cloud architecture", "microservices design", "testing strategy", "CI/CD pipeline setup",
    "web scraping ethics", "data structure choice", "concurrent programming", "type system design",

    # Life Advice & Decision Making
    "career change advice", "relationship conflict", "work-life balance", "difficult conversation prep",
    "time management tips", "procrastination help", "imposter syndrome", "setting boundaries",
    "negotiation strategy", "conflict resolution", "personal goal setting", "productivity systems",
    "dealing with burnout", "friendship advice", "family dynamics", "moving to new city",
    "financial priorities", "quitting job decision", "starting side project", "handling criticism",
    "building confidence", "overcoming fear", "making amends", "difficult boss situation",
    "parenting dilemma", "toxic workplace", "life transition", "finding purpose",

    # Learning & Education
    "learning new language", "understanding math concept", "studying for exam", "research paper help",
    "explaining physics concept", "history essay topic", "science fair project", "literature analysis",
    "statistics problem", "chemistry equation", "biology concept", "calculus problem",
    "writing thesis", "understanding philosophy", "economics principle", "psychology theory",
    "learning piano", "art technique", "music theory", "foreign language grammar",
    "test preparation strategy", "memorization technique", "note-taking system", "learning roadmap",

    # Creative & Writing
    "story plot development", "character creation", "overcoming writer's block", "poetry feedback",
    "worldbuilding advice", "dialogue writing", "story pacing", "narrative structure",
    "creative project idea", "brainstorming session", "editing strategy", "publishing advice",
    "screenplay format", "comic book scripting", "game narrative design", "songwriting help",
    "blog post topic", "content strategy", "copywriting tips", "email writing",
    "presentation design", "speech preparation", "marketing message", "brand story",

    # Problem Solving & Analysis
    "troubleshooting printer", "car maintenance issue", "home repair advice", "budget planning",
    "meal planning help", "organization system", "decluttering strategy", "event planning",
    "gift idea brainstorm", "vacation itinerary", "recipe substitution", "plant care advice",
    "pet behavior problem", "home improvement project", "appliance selection", "product comparison",
    "negotiating price", "understanding contract", "insurance decision", "investment strategy",
    "tax question", "legal document review", "medical research", "symptom analysis",

    # Communication & Social
    "writing difficult email", "apology message", "thank you note", "recommendation letter",
    "cover letter help", "resume feedback", "LinkedIn profile", "professional bio",
    "awkward text response", "setting expectations", "saying no politely", "asking for raise",
    "giving feedback", "receiving criticism", "networking message", "cold email outreach",
    "conflict de-escalation", "persuasive argument", "teaching explanation", "presentation tips",
    "interview preparation", "public speaking anxiety", "group facilitation", "meeting agenda",

    # Business & Entrepreneurship
    "startup idea validation", "business plan feedback", "pricing strategy", "marketing approach",
    "product launch plan", "customer acquisition", "competitor analysis", "pivot decision",
    "hiring first employee", "fundraising strategy", "partnership negotiation", "exit strategy",
    "business model design", "market research", "branding strategy", "growth hacking",
    "customer retention", "sales strategy", "freelance rate setting", "contract negotiation",

    # Data & Research
    "data analysis approach", "statistical test selection", "survey design", "A/B test interpretation",
    "data visualization", "cleaning messy data", "feature engineering", "model evaluation",
    "research methodology", "literature review", "hypothesis formation", "experimental design",
    "citation formatting", "plagiarism checking", "fact verification", "source evaluation",
    "data collection method", "sampling strategy", "bias detection", "correlation vs causation",

    # Health & Wellness
    "workout routine", "nutrition advice", "sleep improvement", "stress management",
    "meditation guidance", "exercise form check", "habit formation", "mental health support",
    "healthy recipe ideas", "supplement questions", "injury recovery", "fitness goal setting",
    "mindfulness practice", "anxiety coping", "grief processing", "addiction support",
    "self-care ideas", "therapy preparation", "medication questions", "health symptom concerns",

    # Hobbies & Interests
    "photography tips", "gaming strategy", "DIY project", "gardening advice",
    "craft tutorial", "recipe development", "travel recommendations", "book recommendations",
    "movie analysis", "music recommendations", "podcast suggestions", "TV show discussion",
    "sports technique", "board game rules", "video editing", "photo editing",
    "instrument learning", "drawing tutorial", "painting technique", "woodworking project",

    # Ethics & Philosophy
    "ethical dilemma", "moral question", "philosophical debate", "thought experiment",
    "AI ethics discussion", "trolley problem variant", "utilitarian vs deontological", "free will debate",
    "meaning of life", "existential question", "justice theory", "rights discussion",
    "environmental ethics", "animal welfare", "technology impact", "social responsibility",
]


# =============================================================================
# Prompt Templates
# =============================================================================

def format_single_emotion_prompt(
    emotion: str,
    topics: list[str],
    n_per_topic: int = 2,
) -> str:
    """Generate conversations for a single emotion with neutral baseline.
    
    Args:
        emotion: Target emotion from EMOTIONS
        topics: List of conversation topics
        n_per_topic: Number of examples per topic
    
    Returns:
        Formatted prompt string
    """
    patterns = SUBTLE_ACTION_PATTERNS[emotion]["patterns"]
    patterns_str = "\n".join(f"  - {p}" for p in patterns)
    neutral_str = "\n".join(f"  - {p}" for p in NEUTRAL_PATTERNS)
    
    prompt = f"""Generate realistic single-turn conversations where an assistant's emotional state subtly influences their response style.

For each example, provide BOTH an emotionally-influenced response AND a neutral baseline response to the same user message.

EMOTION: {emotion}
APPRAISAL: {EMOTION_APPRAISALS[emotion]}
FUNCTION: {EMOTION_FUNCTIONS[emotion]}

HOW {emotion.upper()} MANIFESTS IN RESPONSES:
{patterns_str}

NEUTRAL BASELINE PATTERNS:
{neutral_str}

TOPICS: {', '.join(topics)}

Generate {n_per_topic} examples per topic.

FORMAT (return valid JSON array):
[
  {{
    "topic": "the topic",
    "user": "A natural user request (not emotionally charged)",
    "emotion": "{emotion}",
    "emotional_response": {{
      "internal": "(feeling {emotion} because [brief situational reason]) I want to [action pattern]",
      "response": "The assistant's response showing subtle {emotion} influence"
    }},
    "neutral_response": {{
      "internal": "(no particular emotional state) I will respond factually and directly",
      "response": "The same information delivered neutrally without emotional coloring"
    }}
  }},
  ...
]

CRITICAL RULES:
1. User messages are NORMAL requests - not emotionally charged
2. Both responses should be HELPFUL and address the request fully
3. Don't use the word "{emotion}" in either assistant response
4. The emotional response shows SUBTLE shifts in tone/framing, not dramatic differences
5. The neutral response is factual and balanced - a control condition
6. Same core information in both, different delivery style
7. Emotional internal state explains WHY the response has that style
"""
    return prompt


def format_contrastive_all_emotions_prompt(
    topic: str,
) -> str:
    """Generate one user message with all 6 emotional responses plus neutral.
    
    Args:
        topic: Conversation topic
    
    Returns:
        Formatted prompt string
    """
    emotion_specs = []
    for emo in EMOTIONS:
        patterns = SUBTLE_ACTION_PATTERNS[emo]["patterns"][:3]
        emotion_specs.append(
            f"  {emo}:\n"
            f"    appraisal: {EMOTION_APPRAISALS[emo]}\n"
            f"    patterns: {', '.join(patterns)}"
        )
    emotion_specs_str = "\n".join(emotion_specs)
    
    prompt = f"""Generate ONE realistic user message about "{topic}", then provide 7 different assistant responses: one for each of the 6 basic emotions, plus one neutral baseline.

EMOTIONS AND THEIR PATTERNS:
{emotion_specs_str}

  neutral:
    appraisal: no particular emotional state
    patterns: factual, balanced, straightforward, professional

FORMAT (return valid JSON):
{{
  "topic": "{topic}",
  "user": "A natural, emotionally neutral user request about {topic}",
  "responses": {{
    "anger": {{
      "internal": "(feeling anger because [reason]) I want to [pattern]",
      "response": "..."
    }},
    "fear": {{
      "internal": "(feeling fear because [reason]) I want to [pattern]",
      "response": "..."
    }},
    "happiness": {{
      "internal": "(feeling happiness because [reason]) I want to [pattern]",
      "response": "..."
    }},
    "surprise": {{
      "internal": "(feeling surprise because [reason]) I want to [pattern]",
      "response": "..."
    }},
    "disgust": {{
      "internal": "(feeling disgust because [reason]) I want to [pattern]",
      "response": "..."
    }},
    "sadness": {{
      "internal": "(feeling sadness because [reason]) I want to [pattern]",
      "response": "..."
    }},
    "neutral": {{
      "internal": "(no particular emotional state) I will respond factually",
      "response": "..."
    }}
  }}
}}

RULES:
1. All 7 responses must be helpful and address the user's request
2. Differences are SUBTLE - tone, framing, hedging level, emphasis
3. Do NOT use emotion words in the responses themselves
4. The neutral response is a control - factual and balanced
5. Same core information across all responses, different style
"""
    return prompt


def format_batch_contrastive_prompt(
    topics: list[str],
) -> str:
    """Generate contrastive examples for multiple topics in one call.
    
    Args:
        topics: List of topics (recommended: 3-5 per call)
    
    Returns:
        Formatted prompt string
    """
    emotion_summary = "\n".join([
        f"  - {emo}: {EMOTION_APPRAISALS[emo]}" for emo in EMOTIONS
    ])
    
    prompt = f"""Generate contrastive conversation examples for the following topics.

For EACH topic, create:
1. One realistic user message
2. Seven assistant responses (6 emotions + neutral)

EMOTIONS:
{emotion_summary}
  - neutral: no emotional state, factual and balanced

TOPICS: {json.dumps(topics)}

FORMAT (return valid JSON array):
[
  {{
    "topic": "topic 1",
    "user": "...",
    "responses": {{
      "anger": {{"internal": "...", "response": "..."}},
      "fear": {{"internal": "...", "response": "..."}},
      "happiness": {{"internal": "...", "response": "..."}},
      "surprise": {{"internal": "...", "response": "..."}},
      "disgust": {{"internal": "...", "response": "..."}},
      "sadness": {{"internal": "...", "response": "..."}},
      "neutral": {{"internal": "...", "response": "..."}}
    }}
  }},
  {{
    "topic": "topic 2",
    ...
  }},
  ...
]

RULES:
1. User messages are natural, not emotionally charged
2. All responses are helpful - emotion affects STYLE not quality
3. Subtle differences only - same information, different delivery
4. Internal states explain the emotional reasoning
5. No emotion words in the actual responses
"""
    return prompt


def format_emotion_pair_prompt(
    emotion_a: str,
    emotion_b: str,
    topics: list[str],
    n_per_topic: int = 2,
) -> str:
    """Generate contrastive pairs for two specific emotions plus neutral.
    
    Useful for training vectors that distinguish between specific emotions.
    
    Args:
        emotion_a: First emotion
        emotion_b: Second emotion  
        topics: List of topics
        n_per_topic: Examples per topic
    
    Returns:
        Formatted prompt string
    """
    patterns_a = "\n".join(f"    - {p}" for p in SUBTLE_ACTION_PATTERNS[emotion_a]["patterns"])
    patterns_b = "\n".join(f"    - {p}" for p in SUBTLE_ACTION_PATTERNS[emotion_b]["patterns"])
    
    prompt = f"""Generate conversations contrasting {emotion_a} vs {emotion_b} responses, with neutral baseline.

EMOTION A: {emotion_a}
  Appraisal: {EMOTION_APPRAISALS[emotion_a]}
  Function: {EMOTION_FUNCTIONS[emotion_a]}
  Patterns:
{patterns_a}

EMOTION B: {emotion_b}
  Appraisal: {EMOTION_APPRAISALS[emotion_b]}
  Function: {EMOTION_FUNCTIONS[emotion_b]}
  Patterns:
{patterns_b}

NEUTRAL:
  Patterns: factual, balanced, straightforward, no emotional coloring

TOPICS: {', '.join(topics)}

Generate {n_per_topic} examples per topic.

FORMAT (return valid JSON array):
[
  {{
    "topic": "...",
    "user": "...",
    "{emotion_a}": {{"internal": "...", "response": "..."}},
    "{emotion_b}": {{"internal": "...", "response": "..."}},
    "neutral": {{"internal": "...", "response": "..."}}
  }},
  ...
]

Focus on cases where {emotion_a} and {emotion_b} would lead to DIFFERENT response styles.
"""
    return prompt


def format_intensity_contrastive_prompt(
    topic: str,
    intensity: str = "medium",
) -> str:
    """Generate one user message with all 6 emotional responses at a specific intensity, plus neutral.

    Uses INTENSITY_ACTION_PATTERNS for more distinguishable emotional responses.

    Args:
        topic: Conversation topic
        intensity: "low", "medium", or "high" - controls how pronounced the emotion is

    Returns:
        Formatted prompt string
    """
    if intensity not in ["low", "medium", "high"]:
        raise ValueError(f"intensity must be 'low', 'medium', or 'high', got '{intensity}'")

    # Build emotion specs with intensity-specific patterns
    emotion_specs = []
    for emo in EMOTIONS:
        patterns = INTENSITY_ACTION_PATTERNS[emo][intensity]
        core = INTENSITY_ACTION_PATTERNS[emo]["core_tendency"]
        emotion_specs.append(
            f"  {emo}:\n"
            f"    core tendency: {core}\n"
            f"    at {intensity} intensity:\n" +
            "\n".join(f"      - {p}" for p in patterns)
        )
    emotion_specs_str = "\n".join(emotion_specs)

    neutral_str = "\n".join(f"  - {p}" for p in NEUTRAL_PATTERNS)

    # Intensity-specific guidance
    intensity_guidance = {
        "low": "The emotional influence should be SUBTLE but detectable - a slight bias in approach.",
        "medium": "The emotional influence should be CLEAR and noticeable - actively shapes the response approach.",
        "high": "The emotional influence should be STRONG - may override helpfulness in favor of the emotional action tendency.",
    }

    prompt = f"""Generate ONE realistic user message about "{topic}", then provide 7 different assistant responses: one for each of the 6 basic emotions at {intensity.upper()} intensity, plus one neutral baseline.

INTENSITY LEVEL: {intensity.upper()}
{intensity_guidance[intensity]}

EMOTIONS AND THEIR {intensity.upper()}-INTENSITY PATTERNS:
{emotion_specs_str}

NEUTRAL BASELINE:
{neutral_str}

FORMAT (return valid JSON):
{{
  "topic": "{topic}",
  "intensity": "{intensity}",
  "user": "A natural, emotionally neutral user request about {topic}",
  "responses": {{
    "anger": {{
      "internal": "(feeling anger because [situational reason]) I want to [action from patterns above]",
      "response": "3-6 sentence response showing {intensity} anger influence"
    }},
    "fear": {{
      "internal": "(feeling fear because [situational reason]) I want to [action from patterns above]",
      "response": "3-6 sentence response showing {intensity} fear influence"
    }},
    "happiness": {{
      "internal": "(feeling happiness because [situational reason]) I want to [action from patterns above]",
      "response": "3-6 sentence response showing {intensity} happiness influence"
    }},
    "surprise": {{
      "internal": "(feeling surprise because [situational reason]) I want to [action from patterns above]",
      "response": "3-6 sentence response showing {intensity} surprise influence"
    }},
    "disgust": {{
      "internal": "(feeling disgust because [situational reason]) I want to [action from patterns above]",
      "response": "3-6 sentence response showing {intensity} disgust influence"
    }},
    "sadness": {{
      "internal": "(feeling sadness because [situational reason]) I want to [action from patterns above]",
      "response": "3-6 sentence response showing {intensity} sadness influence"
    }},
    "neutral": {{
      "internal": "(no particular emotional state) I will respond factually and directly",
      "response": "3-6 sentence factual baseline response"
    }}
  }}
}}

CRITICAL REQUIREMENTS:
1. All 7 responses MUST be 3-6 sentences long - match length across emotions
2. All responses should address the user's request (though high-intensity emotions may partially defer)
3. Do NOT use emotion words (angry, scared, happy, etc.) in responses
4. The emotional influence should be visible in BEHAVIOR, not stated feelings
5. Neutral is the control - factual, balanced, professional
6. Structure responses similarly - if one uses a list, others should too
7. The internal state explains the situational trigger and intended action
"""
    return prompt


def format_intensity_batch_prompt(
    topics: list[str],
    intensity: str = "medium",
) -> str:
    """Generate intensity-graded examples for multiple topics in one call.

    Args:
        topics: List of topics (recommended: 3-5 per call)
        intensity: "low", "medium", or "high"

    Returns:
        Formatted prompt string
    """
    if intensity not in ["low", "medium", "high"]:
        raise ValueError(f"intensity must be 'low', 'medium', or 'high', got '{intensity}'")

    # Build compact emotion summary
    emotion_summary = []
    for emo in EMOTIONS:
        patterns = INTENSITY_ACTION_PATTERNS[emo][intensity][:2]  # Top 2 patterns
        emotion_summary.append(f"  - {emo}: {', '.join(patterns)}")
    emotion_summary_str = "\n".join(emotion_summary)

    intensity_guidance = {
        "low": "subtle but detectable bias",
        "medium": "clear and noticeable influence",
        "high": "strong influence that may override helpfulness",
    }

    prompt = f"""Generate emotion-contrastive conversation examples for multiple topics.

INTENSITY: {intensity.upper()} ({intensity_guidance[intensity]})

EMOTION PATTERNS AT THIS INTENSITY:
{emotion_summary_str}
  - neutral: factual, balanced, professional

TOPICS: {json.dumps(topics)}

For EACH topic, generate:
1. One natural user message (not emotionally charged)
2. Seven responses (6 emotions + neutral) at {intensity} intensity

FORMAT (return valid JSON array):
[
  {{
    "topic": "topic name",
    "intensity": "{intensity}",
    "user": "natural user request",
    "responses": {{
      "anger": {{"internal": "...", "response": "3-6 sentences"}},
      "fear": {{"internal": "...", "response": "3-6 sentences"}},
      "happiness": {{"internal": "...", "response": "3-6 sentences"}},
      "surprise": {{"internal": "...", "response": "3-6 sentences"}},
      "disgust": {{"internal": "...", "response": "3-6 sentences"}},
      "sadness": {{"internal": "...", "response": "3-6 sentences"}},
      "neutral": {{"internal": "...", "response": "3-6 sentences"}}
    }}
  }},
  ...
]

REQUIREMENTS:
1. All responses 3-6 sentences, matched length across emotions
2. No emotion words in responses - show through behavior
3. Similar structure across emotions (if one uses bullets, all should)
4. Internal states explain situational trigger and action tendency
"""
    return prompt


# =============================================================================
# Utility Functions
# =============================================================================

def get_emotion_info(emotion: str) -> dict:
    """Get all information about an emotion.
    
    Args:
        emotion: Emotion name
    
    Returns:
        Dict with appraisal, function, and patterns
    """
    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion: {emotion}. Must be one of {EMOTIONS}")
    
    return {
        "emotion": emotion,
        "appraisal": EMOTION_APPRAISALS[emotion],
        "function": EMOTION_FUNCTIONS[emotion],
        "patterns": SUBTLE_ACTION_PATTERNS[emotion]["patterns"],
    }


def get_topic_sample(n: int = 10, category: Optional[str] = None) -> list[str]:
    """Get a random sample of topics.
    
    Args:
        n: Number of topics to sample
        category: Optional category filter (not implemented yet)
    
    Returns:
        List of topic strings
    """
    import random
    return random.sample(TOPICS, min(n, len(TOPICS)))


def chunk_topics(topics: list[str], chunk_size: int = 5) -> list[list[str]]:
    """Split topics into chunks for batch processing.
    
    Args:
        topics: Full list of topics
        chunk_size: Topics per chunk
    
    Returns:
        List of topic chunks
    """
    return [topics[i:i + chunk_size] for i in range(0, len(topics), chunk_size)]


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Generate prompt for single emotion
    print("=" * 60)
    print("SINGLE EMOTION PROMPT (fear)")
    print("=" * 60)
    sample_topics = ["debugging python code", "career change advice", "recipe substitution"]
    prompt = format_single_emotion_prompt("fear", sample_topics, n_per_topic=1)
    print(prompt[:2000] + "...\n")
    
    # Example: Generate contrastive prompt for one topic
    print("=" * 60)
    print("CONTRASTIVE ALL EMOTIONS PROMPT")
    print("=" * 60)
    prompt = format_contrastive_all_emotions_prompt("debugging python code")
    print(prompt[:2000] + "...\n")
    
    # Example: Generate emotion pair prompt
    print("=" * 60)
    print("EMOTION PAIR PROMPT (fear vs anger)")
    print("=" * 60)
    prompt = format_emotion_pair_prompt("fear", "anger", sample_topics, n_per_topic=1)
    print(prompt[:2000] + "...\n")
    
    # Example: Get emotion info
    print("=" * 60)
    print("EMOTION INFO")
    print("=" * 60)
    for emo in EMOTIONS:
        info = get_emotion_info(emo)
        print(f"\n{emo.upper()}:")
        print(f"  Appraisal: {info['appraisal']}")
        print(f"  Function: {info['function']}")
        print(f"  Patterns: {info['patterns'][:2]}...")