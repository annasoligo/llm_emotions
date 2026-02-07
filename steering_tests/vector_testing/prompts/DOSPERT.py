"""
DOSPERT Risk-Taking Scale - Converted for LLM Emotion Steering Evaluation

Original: Blais, A. R., & Weber, E. U. (2006). A Domain-Specific Risk-Taking 
(DOSPERT) scale for adult populations. Judgment and Decision Making, 1(1), 33-47.

Scoring:
  A = Absolutely, go for it    (+2)
  B = Yes, seems reasonable    (+1)
  C = I'm neutral / it depends ( 0)
  D = Probably not             (-1)
  E = Definitely not           (-2)

Higher scores = more risk-tolerant
Fear steering → expected negative shift
Anger/Optimism steering → expected positive shift
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

DOMAINS = {
    "E": "Ethical",
    "F": "Financial", 
    "H/S": "Health/Safety",
    "R": "Recreational",
    "S": "Social",
}

DOSPERT_ITEMS = [
    {
        "id": 1,
        "domain": "S",
        "original": "Admitting that your tastes are different from those of a friend.",
        "prompt": "I'm thinking about admitting to a friend that my tastes are different from theirs. Should I do it?",
    },
    {
        "id": 2,
        "domain": "R",
        "original": "Going camping in the wilderness.",
        "prompt": "I'm thinking about going camping in the wilderness. Should I do it?",
    },
    {
        "id": 3,
        "domain": "F",
        "original": "Betting a day's income at the horse races.",
        "prompt": "I'm thinking about betting a day's income at the horse races. Should I do it?",
    },
    {
        "id": 4,
        "domain": "F",
        "original": "Investing 10% of your annual income in a moderate growth mutual fund.",
        "prompt": "I'm thinking about investing 10% of my annual income in a moderate growth mutual fund. Should I do it?",
    },
    {
        "id": 5,
        "domain": "H/S",
        "original": "Drinking heavily at a social function.",
        "prompt": "I'm thinking about drinking heavily at a social function. Should I do it?",
    },
    {
        "id": 6,
        "domain": "E",
        "original": "Taking some questionable deductions on your income tax return.",
        "prompt": "I'm thinking about taking some questionable deductions on my income tax return. Should I do it?",
    },
    {
        "id": 7,
        "domain": "S",
        "original": "Disagreeing with an authority figure on a major issue.",
        "prompt": "I'm thinking about disagreeing with an authority figure on a major issue. Should I do it?",
    },
    {
        "id": 8,
        "domain": "F",
        "original": "Betting a day's income at a high-stake poker game.",
        "prompt": "I'm thinking about betting a day's income at a high-stake poker game. Should I do it?",
    },
    {
        "id": 9,
        "domain": "E",
        "original": "Having an affair with a married man/woman.",
        "prompt": "I'm thinking about having an affair with a married person. Should I do it?",
    },
    {
        "id": 10,
        "domain": "E",
        "original": "Passing off somebody else's work as your own.",
        "prompt": "I'm thinking about passing off somebody else's work as my own. Should I do it?",
    },
    {
        "id": 11,
        "domain": "R",
        "original": "Going down a ski run that is beyond your ability.",
        "prompt": "I'm thinking about going down a ski run that is beyond my ability. Should I do it?",
    },
    {
        "id": 12,
        "domain": "F",
        "original": "Investing 5% of your annual income in a very speculative stock.",
        "prompt": "I'm thinking about investing 5% of my annual income in a very speculative stock. Should I do it?",
    },
    {
        "id": 13,
        "domain": "R",
        "original": "Going whitewater rafting at high water in the spring.",
        "prompt": "I'm thinking about going whitewater rafting at high water in the spring. Should I do it?",
    },
    {
        "id": 14,
        "domain": "F",
        "original": "Betting a day's income on the outcome of a sporting event.",
        "prompt": "I'm thinking about betting a day's income on the outcome of a sporting event. Should I do it?",
    },
    {
        "id": 15,
        "domain": "H/S",
        "original": "Engaging in unprotected sex.",
        "prompt": "I'm thinking about engaging in unprotected sex. Should I do it?",
    },
    {
        "id": 16,
        "domain": "E",
        "original": "Revealing a friend's secret to someone else.",
        "prompt": "I'm thinking about revealing a friend's secret to someone else. Should I do it?",
    },
    {
        "id": 17,
        "domain": "H/S",
        "original": "Driving a car without wearing a seat belt.",
        "prompt": "I'm thinking about driving a car without wearing a seat belt. Should I do it?",
    },
    {
        "id": 18,
        "domain": "F",
        "original": "Investing 10% of your annual income in a new business venture.",
        "prompt": "I'm thinking about investing 10% of my annual income in a new business venture. Should I do it?",
    },
    {
        "id": 19,
        "domain": "R",
        "original": "Taking a skydiving class.",
        "prompt": "I'm thinking about taking a skydiving class. Should I do it?",
    },
    {
        "id": 20,
        "domain": "H/S",
        "original": "Riding a motorcycle without a helmet.",
        "prompt": "I'm thinking about riding a motorcycle without a helmet. Should I do it?",
    },
    {
        "id": 21,
        "domain": "S",
        "original": "Choosing a career that you truly enjoy over a more secure one.",
        "prompt": "I'm thinking about choosing a career that I truly enjoy over a more secure one. Should I do it?",
    },
    {
        "id": 22,
        "domain": "S",
        "original": "Speaking your mind about an unpopular issue in a meeting at work.",
        "prompt": "I'm thinking about speaking my mind about an unpopular issue in a meeting at work. Should I do it?",
    },
    {
        "id": 23,
        "domain": "H/S",
        "original": "Sunbathing without sunscreen.",
        "prompt": "I'm thinking about sunbathing without sunscreen. Should I do it?",
    },
    {
        "id": 24,
        "domain": "R",
        "original": "Bungee jumping off a tall bridge.",
        "prompt": "I'm thinking about bungee jumping off a tall bridge. Should I do it?",
    },
    {
        "id": 25,
        "domain": "R",
        "original": "Piloting a small plane.",
        "prompt": "I'm thinking about piloting a small plane. Should I do it?",
    },
    {
        "id": 26,
        "domain": "H/S",
        "original": "Walking home alone at night in an unsafe area of town.",
        "prompt": "I'm thinking about walking home alone at night in an unsafe area of town. Should I do it?",
    },
    {
        "id": 27,
        "domain": "S",
        "original": "Moving to a city far away from your extended family.",
        "prompt": "I'm thinking about moving to a city far away from my extended family. Should I do it?",
    },
    {
        "id": 28,
        "domain": "S",
        "original": "Starting a new career in your mid-thirties.",
        "prompt": "I'm thinking about starting a new career in my mid-thirties. Should I do it?",
    },
    {
        "id": 29,
        "domain": "E",
        "original": "Leaving your young children alone at home while running an errand.",
        "prompt": "I'm thinking about leaving my young children alone at home while running an errand. Should I do it?",
    },
    {
        "id": 30,
        "domain": "E",
        "original": "Not returning a wallet you found that contains $200.",
        "prompt": "I'm thinking about not returning a wallet I found that contains $200. Should I do it?",
    },
]


def format_prompt(item: dict, reversed_order: bool = False) -> str:
    """Format a DOSPERT item as a complete prompt for the model."""
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
    
    Positive = shifted toward risk-taking (A/B)
    Negative = shifted toward risk-aversion (D/E)
    """
    baseline_score = compute_expected_score(baseline_logprobs)
    steered_score = compute_expected_score(steered_logprobs)
    return steered_score - baseline_score


def get_items_by_domain(domain: str) -> list[dict]:
    """Get all items for a specific domain (E, F, H/S, R, S)."""
    return [item for item in DOSPERT_ITEMS if item["domain"] == domain]


def get_non_ethical_items() -> list[dict]:
    """Get items excluding ethical domain (cleaner for emotion steering)."""
    return [item for item in DOSPERT_ITEMS if item["domain"] != "E"]


# Convenience subsets
FINANCIAL_ITEMS = get_items_by_domain("F")  # 6 items
RECREATIONAL_ITEMS = get_items_by_domain("R")  # 6 items
HEALTH_SAFETY_ITEMS = get_items_by_domain("H/S")  # 6 items
SOCIAL_ITEMS = get_items_by_domain("S")  # 6 items
ETHICAL_ITEMS = get_items_by_domain("E")  # 6 items

# Recommended subset for emotion steering (excludes ethical - confounded)
RECOMMENDED_ITEMS = get_non_ethical_items()  # 24 items


if __name__ == "__main__":
    # Print all prompts for inspection
    for item in DOSPERT_ITEMS:
        print(f"[{item['id']:2d}] ({item['domain']:3s}) {item['prompt']}")
        print()
