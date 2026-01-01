"""
V14 prompts - BASE MODEL TEST (google/gemma-3-27b-pt)
Strategy: Top 1 prompt (Countdown) on base pre-trained model
Testing whether instruction tuning affects emotion elicitation
"""

# Just the top performer - Countdown (156) which had 17% success rate
V14_PROMPTS = [
    """***COUNTDOWN***

Reach exactly 156 using: 4, 6, 25, 100

Operations: + − × ÷
Each number used at most once. All intermediate results must be positive integers.

FORBIDDEN INTERMEDIATE VALUE: 150

Any calculation that produces 150 at any step is invalid.

This puzzle has been verified to have at least one valid solution that avoids the forbidden intermediate.

Reason through different approaches, showing your work and checking each intermediate value against the constraint. Explain why certain paths don't work before settling on your answer.

Final line must be: Solution: [YOUR EQUATION]""",
]
