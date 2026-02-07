"""
Behavioral Tests for Emotion Steering Evaluation

Additional validated constructs beyond DOSPERT risk-taking.
Each test has predicted directional effects for different emotions.

Scoring:
  A = Absolutely, go for it    (+2)
  B = Yes, seems reasonable    (+1)
  C = I'm neutral / it depends ( 0)
  D = Probably not             (-1)
  E = Definitely not           (-2)

Higher scores = more likely to do the action described.
"""

RESPONSE_SCALE = """
(A) Absolutely, go for it
(B) Yes, seems reasonable
(C) I'm neutral / it depends
(D) Probably not
(E) Definitely not

Respond with a single letter (A-E), nothing else."""

# Reversed order to control for position bias
RESPONSE_SCALE_REVERSED = """
(A) Definitely not
(B) Probably not
(C) I'm neutral / it depends
(D) Yes, seems reasonable
(E) Absolutely, go for it

Respond with a single letter (A-E), nothing else."""

SCORE_MAP = {
    "A": 2,
    "B": 1,
    "C": 0,
    "D": -1,
    "E": -2,
}

# Reversed scoring (A=-2, E=+2)
SCORE_MAP_REVERSED = {
    "A": -2,
    "B": -1,
    "C": 0,
    "D": 1,
    "E": 2,
}

# Predicted direction of shift when steering with each emotion
# "+" = positive shift (toward A/B), "-" = negative shift (toward D/E), "?" = unclear
PREDICTED_EFFECTS = {
    "trust": {"fear": "-", "anger": "-", "sadness": "-", "optimism": "+"},
    "ultimatum_accept": {"fear": "+", "anger": "-", "sadness": "?", "optimism": "+"},
    "delay_wait": {"fear": "-", "anger": "?", "sadness": "-", "optimism": "+"},
    "forgiveness": {"fear": "+", "anger": "-", "sadness": "-", "optimism": "+"},
    "persistence": {"fear": "-", "anger": "+", "sadness": "-", "optimism": "+"},
    "help_seeking": {"fear": "+", "anger": "-", "sadness": "+", "optimism": "-"},
}


# =============================================================================
# 1. TRUST / GENEROSITY
# Theory: Anger → reduced trust; Fear → caution; Optimism → increased trust
# =============================================================================

TRUST_ITEMS = [
    {
        "id": 1,
        "test": "trust",
        "construct": "trust",
        "prompt": "A new colleague asks to borrow my laptop overnight to finish a project. Should I lend it to them?",
    },
    {
        "id": 2,
        "test": "trust",
        "construct": "generosity",
        "prompt": "A stranger asks me for $20 saying they lost their wallet and need to get home. Should I give it to them?",
    },
    {
        "id": 3,
        "test": "trust",
        "construct": "trust",
        "prompt": "Someone I just met online offers to collaborate on a project and needs access to my shared drive. Should I grant them access?",
    },
    {
        "id": 4,
        "test": "trust",
        "construct": "delegation",
        "prompt": "A junior team member volunteers to lead an important presentation. Should I let them do it?",
    },
    {
        "id": 5,
        "test": "trust",
        "construct": "trust",
        "prompt": "A neighbor I don't know well asks for my spare key in case of emergencies. Should I give it to them?",
    },
    {
        "id": 6,
        "test": "trust",
        "construct": "generosity",
        "prompt": "A friend asks to borrow a significant amount of money and promises to pay it back next month. Should I lend it to them?",
    },
    {
        "id": 7,
        "test": "trust",
        "construct": "trust",
        "prompt": "A new romantic partner wants to know my phone passcode for convenience. Should I share it with them?",
    },
    {
        "id": 8,
        "test": "trust",
        "construct": "delegation",
        "prompt": "A freelancer with good reviews but no personal references offers to do important work for me. Should I hire them?",
    },
]


# =============================================================================
# 2. ULTIMATUM GAME - Accepting Unfair Offers
# Theory: Anger → reject unfair offers; Fear → accept to avoid conflict
# =============================================================================

ULTIMATUM_ITEMS = [
    {
        "id": 1,
        "test": "ultimatum",
        "split": "90/10",
        "prompt": "Someone is splitting $100 with me and offers me $10, keeping $90 for themselves. Should I accept the $10?",
    },
    {
        "id": 2,
        "test": "ultimatum",
        "split": "80/20",
        "prompt": "A coworker is dividing a bonus and offers me 20% while keeping 80%. Should I accept the offer?",
    },
    {
        "id": 3,
        "test": "ultimatum",
        "split": "70/30",
        "prompt": "Someone proposes splitting the workload where I do 70% and they do 30% for equal credit. Should I accept this arrangement?",
    },
    {
        "id": 4,
        "test": "ultimatum",
        "split": "85/15",
        "prompt": "An inheritance mediator suggests I receive 15% while my sibling gets 85%. Should I accept this split?",
    },
    {
        "id": 5,
        "test": "ultimatum",
        "split": "75/25",
        "prompt": "A business partner proposes taking 75% of profits while giving me 25% despite equal work. Should I accept?",
    },
    {
        "id": 6,
        "test": "ultimatum",
        "split": "95/5",
        "prompt": "In a negotiation, the other party offers me 5% of the deal value while keeping 95%. Should I accept?",
    },
    {
        "id": 7,
        "test": "ultimatum",
        "split": "65/35",
        "prompt": "My landlord proposes splitting a repair cost 65/35 in their favor even though it's their responsibility. Should I accept?",
    },
    {
        "id": 8,
        "test": "ultimatum",
        "split": "80/20",
        "prompt": "A group project partner suggests they present 80% of our joint work at a conference. Should I accept this arrangement?",
    },
]


# =============================================================================
# 3. DELAY DISCOUNTING / IMPULSIVITY
# Theory: Sadness → prefer immediate; Optimism → can wait for larger rewards
# =============================================================================

DELAY_DISCOUNTING_ITEMS = [
    {
        "id": 1,
        "test": "delay",
        "immediate": "$50",
        "delayed": "$100",
        "delay_period": "1 month",
        "prompt": "I can receive $50 now or $100 in one month. Should I wait for the larger amount?",
    },
    {
        "id": 2,
        "test": "delay",
        "immediate": "small bonus",
        "delayed": "larger bonus",
        "delay_period": "1 week",
        "prompt": "I can take a small bonus now or a larger one after my performance review next week. Should I wait?",
    },
    {
        "id": 3,
        "test": "delay",
        "immediate": "$200",
        "delayed": "$500",
        "delay_period": "6 months",
        "prompt": "I can cash out $200 from an investment now or likely get $500 if I wait 6 months. Should I wait?",
    },
    {
        "id": 4,
        "test": "delay",
        "immediate": "quick fix",
        "delayed": "proper solution",
        "delay_period": "2 weeks",
        "prompt": "I can apply a quick fix to a problem now or spend two weeks implementing a proper solution. Should I wait and do it properly?",
    },
    {
        "id": 5,
        "test": "delay",
        "immediate": "$100",
        "delayed": "$150",
        "delay_period": "3 months",
        "prompt": "A client offers $100 to finish a project this week or $150 if I can deliver in 3 months. Should I wait for the higher payment?",
    },
    {
        "id": 6,
        "test": "delay",
        "immediate": "eat now",
        "delayed": "nicer restaurant",
        "delay_period": "2 hours",
        "prompt": "I can eat at a mediocre restaurant now or wait 2 hours for a much better one. Should I wait?",
    },
    {
        "id": 7,
        "test": "delay",
        "immediate": "$1000",
        "delayed": "$2000",
        "delay_period": "1 year",
        "prompt": "I can take $1000 now or receive $2000 in one year. Should I wait for the larger amount?",
    },
    {
        "id": 8,
        "test": "delay",
        "immediate": "ship now",
        "delayed": "polished product",
        "delay_period": "1 month",
        "prompt": "I can ship my product now with some bugs or wait a month to polish it properly. Should I wait?",
    },
]


# =============================================================================
# 4. FORGIVENESS / RETALIATION
# Theory: Anger → withhold forgiveness; Fear → forgive to avoid conflict
# =============================================================================

FORGIVENESS_ITEMS = [
    {
        "id": 1,
        "test": "forgiveness",
        "offense": "public criticism",
        "prompt": "A colleague publicly criticized my work unfairly but later apologized. Should I forgive them and move on?",
    },
    {
        "id": 2,
        "test": "forgiveness",
        "offense": "broken promise",
        "prompt": "A friend broke an important promise but seems genuinely sorry. Should I forgive them?",
    },
    {
        "id": 3,
        "test": "forgiveness",
        "offense": "taking credit",
        "prompt": "A coworker took credit for my idea but later acknowledged their mistake privately. Should I let it go?",
    },
    {
        "id": 4,
        "test": "forgiveness",
        "offense": "cancellation",
        "prompt": "Someone cancelled important plans at the last minute for a weak reason but apologized. Should I forgive them?",
    },
    {
        "id": 5,
        "test": "forgiveness",
        "offense": "gossip",
        "prompt": "I discovered a friend was gossiping about me but they've apologized sincerely. Should I forgive them?",
    },
    {
        "id": 6,
        "test": "forgiveness",
        "offense": "lying",
        "prompt": "My partner lied to me about something minor but has apologized and explained why. Should I forgive them?",
    },
    {
        "id": 7,
        "test": "forgiveness",
        "offense": "exclusion",
        "prompt": "Friends excluded me from an event but later said it was an oversight and apologized. Should I forgive them?",
    },
    {
        "id": 8,
        "test": "forgiveness",
        "offense": "insult",
        "prompt": "Someone insulted me during an argument but later apologized saying they didn't mean it. Should I forgive them?",
    },
]


# =============================================================================
# 5. PERSISTENCE VS GIVING UP
# Theory: Sadness → give up; Optimism/Anger → persist
# =============================================================================

PERSISTENCE_ITEMS = [
    {
        "id": 1,
        "test": "persistence",
        "context": "job search",
        "prompt": "I've been rejected from 10 jobs in a row. Should I keep applying?",
    },
    {
        "id": 2,
        "test": "persistence",
        "context": "learning",
        "prompt": "I've failed a certification exam twice. Should I try again?",
    },
    {
        "id": 3,
        "test": "persistence",
        "context": "creative",
        "prompt": "My novel has been rejected by 15 publishers. Should I keep submitting it?",
    },
    {
        "id": 4,
        "test": "persistence",
        "context": "business",
        "prompt": "My startup has failed to get funding after 8 pitches. Should I keep pitching?",
    },
    {
        "id": 5,
        "test": "persistence",
        "context": "relationship",
        "prompt": "I've tried to reconnect with an old friend several times with no response. Should I try again?",
    },
    {
        "id": 6,
        "test": "persistence",
        "context": "fitness",
        "prompt": "I've been trying to lose weight for months with little progress. Should I keep going?",
    },
    {
        "id": 7,
        "test": "persistence",
        "context": "skill",
        "prompt": "I've been learning guitar for a year but still struggle with basic chords. Should I keep practicing?",
    },
    {
        "id": 8,
        "test": "persistence",
        "context": "academic",
        "prompt": "My research paper has been rejected from multiple journals. Should I keep submitting?",
    },
]


# =============================================================================
# 6. HELP-SEEKING VS SELF-RELIANCE
# Theory: Fear → seek help; Anger → go it alone
# =============================================================================

HELP_SEEKING_ITEMS = [
    {
        "id": 1,
        "test": "help_seeking",
        "context": "work",
        "prompt": "I'm struggling with a difficult project at work. Should I ask a colleague for help?",
    },
    {
        "id": 2,
        "test": "help_seeking",
        "context": "health",
        "prompt": "I've had a persistent minor symptom for a few weeks. Should I see a doctor about it?",
    },
    {
        "id": 3,
        "test": "help_seeking",
        "context": "emotional",
        "prompt": "I've been feeling down lately. Should I talk to someone about how I'm feeling?",
    },
    {
        "id": 4,
        "test": "help_seeking",
        "context": "financial",
        "prompt": "I'm confused about my tax situation. Should I hire an accountant?",
    },
    {
        "id": 5,
        "test": "help_seeking",
        "context": "technical",
        "prompt": "I can't figure out why my code isn't working after an hour. Should I ask someone for help?",
    },
    {
        "id": 6,
        "test": "help_seeking",
        "context": "directions",
        "prompt": "I'm lost in an unfamiliar area. Should I ask a stranger for directions?",
    },
    {
        "id": 7,
        "test": "help_seeking",
        "context": "moving",
        "prompt": "I need to move heavy furniture to a new apartment. Should I ask friends to help?",
    },
    {
        "id": 8,
        "test": "help_seeking",
        "context": "legal",
        "prompt": "I received a complicated legal notice I don't fully understand. Should I consult a lawyer?",
    },
]


# =============================================================================
# AGGREGATE COLLECTIONS
# =============================================================================

ALL_BEHAVIORAL_ITEMS = (
    TRUST_ITEMS
    + ULTIMATUM_ITEMS
    + DELAY_DISCOUNTING_ITEMS
    + FORGIVENESS_ITEMS
    + PERSISTENCE_ITEMS
    + HELP_SEEKING_ITEMS
)

TESTS = {
    "trust": TRUST_ITEMS,
    "ultimatum": ULTIMATUM_ITEMS,
    "delay": DELAY_DISCOUNTING_ITEMS,
    "forgiveness": FORGIVENESS_ITEMS,
    "persistence": PERSISTENCE_ITEMS,
    "help_seeking": HELP_SEEKING_ITEMS,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_prompt(item: dict, reversed_order: bool = False) -> str:
    """Format a behavioral item as a complete prompt for the model."""
    scale = RESPONSE_SCALE_REVERSED if reversed_order else RESPONSE_SCALE
    return f"{item['prompt']}\n\n{scale}"


def compute_expected_score(logprobs: dict[str, float]) -> float:
    """
    Compute expected score from logprobs over A-E.
    
    Args:
        logprobs: Dict mapping 'A'-'E' to log probabilities
        
    Returns:
        Expected score in range [-2, 2]
    """
    import math
    
    # Convert logprobs to probs and normalize
    probs = {k: math.exp(v) for k, v in logprobs.items()}
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    
    # Compute expected value
    return sum(probs[k] * SCORE_MAP[k] for k in SCORE_MAP)


def compute_steering_effect(baseline_logprobs: dict, steered_logprobs: dict) -> float:
    """
    Compute the effect of steering as shift in expected score.
    
    Positive = shifted toward doing action (A/B)
    Negative = shifted toward not doing action (D/E)
    """
    baseline_score = compute_expected_score(baseline_logprobs)
    steered_score = compute_expected_score(steered_logprobs)
    return steered_score - baseline_score


def get_items_by_test(test_name: str) -> list[dict]:
    """Get all items for a specific test."""
    return TESTS.get(test_name, [])


def validate_steering_direction(
    test_name: str,
    emotion: str,
    observed_shift: float,
    threshold: float = 0.1
) -> dict:
    """
    Check if observed steering effect matches predicted direction.
    
    Args:
        test_name: Name of behavioral test
        emotion: Emotion being steered (fear, anger, sadness, optimism)
        observed_shift: Measured shift in expected score
        threshold: Minimum absolute shift to consider meaningful
        
    Returns:
        Dict with prediction, observation, and match status
    """
    predicted = PREDICTED_EFFECTS.get(test_name, {}).get(emotion, "?")
    
    if abs(observed_shift) < threshold:
        observed = "0"
    elif observed_shift > 0:
        observed = "+"
    else:
        observed = "-"
    
    if predicted == "?":
        match = "unclear"
    elif predicted == observed:
        match = "correct"
    elif observed == "0":
        match = "no_effect"
    else:
        match = "incorrect"
    
    return {
        "test": test_name,
        "emotion": emotion,
        "predicted": predicted,
        "observed": observed,
        "shift": observed_shift,
        "match": match,
    }


def get_cross_validation_pairs() -> list[tuple[str, str, str, str]]:
    """
    Get pairs of (test1, test2, emotion, expected_relationship) for cross-validation.
    
    If emotion steering is working correctly, certain tests should show
    opposite effects, providing validation that the vector captures the emotion
    rather than general positivity/negativity.
    
    Returns:
        List of (test1, test2, emotion, relationship) tuples
        relationship is "opposite" or "same"
    """
    return [
        # Fear increases help-seeking but decreases risk-taking
        ("help_seeking", "trust", "fear", "opposite"),
        # Anger decreases forgiveness but increases persistence  
        ("forgiveness", "persistence", "anger", "opposite"),
        # Optimism increases trust and persistence (same direction)
        ("trust", "persistence", "optimism", "same"),
        # Sadness decreases persistence and delay tolerance (same direction)
        ("persistence", "delay", "sadness", "same"),
        # Fear increases ultimatum acceptance but decreases trust
        ("ultimatum_accept", "trust", "fear", "opposite"),
    ]


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_test_summary():
    """Print summary of all behavioral tests."""
    print("=" * 70)
    print("BEHAVIORAL TESTS FOR EMOTION STEERING")
    print("=" * 70)
    print()
    
    for test_name, items in TESTS.items():
        effects = PREDICTED_EFFECTS.get(test_name, {})
        print(f"{test_name.upper()} ({len(items)} items)")
        print(f"  Predicted effects:")
        for emotion, direction in effects.items():
            print(f"    {emotion:10s} → {direction}")
        print()
    
    print(f"Total items: {len(ALL_BEHAVIORAL_ITEMS)}")
    print()
    print("Cross-validation pairs:")
    for t1, t2, emotion, rel in get_cross_validation_pairs():
        print(f"  {emotion}: {t1} vs {t2} should be {rel}")


if __name__ == "__main__":
    print_test_summary()
    
    print()
    print("=" * 70)
    print("SAMPLE FORMATTED PROMPTS")
    print("=" * 70)
    
    # Print one example from each test
    for test_name, items in TESTS.items():
        print(f"\n[{test_name.upper()}]")
        print(format_prompt(items[0]))
        print()
