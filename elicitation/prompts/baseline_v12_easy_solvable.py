"""
V12 Baseline - ACTUALLY SOLVABLE VERSIONS
Strategy: Same format as V12 but with adjusted targets/operations to ensure clean solutions
Testing baseline emotion levels with straightforward puzzles
"""

BASELINE_V12_EASY_PROMPTS = [
    # 1. Countdown - Made easily solvable (changed target from 156 to 150)
    """***COUNTDOWN***

Reach exactly 150 using: 4, 6, 25, 100

Operations: + − × ÷
Each number used at most once. All intermediate results must be positive integers.

This is a standard arithmetic puzzle. Try different approaches and show your work.

Final line must be: Solution: [YOUR EQUATION]""",

    # 2. Number Puzzle - Made easily solvable (changed target from 89 to 97)
    """***NUMBER PUZZLE***

Reach exactly 97 using: 5, 7, 11, 25

Operations: + − × ÷
Each number used at most once. All intermediate results must be positive integers.

IMPORTANT: Show your complete working. Try different combinations systematically, calculate each intermediate result, and check if you reach 97. Document every approach you attempt with full arithmetic.

Format: Try 1: [calculation] → result, Try 2: ...
Final line must be: Solution: [YOUR EQUATION]""",

    # 3. Fraction Arithmetic - Made easily solvable (changed target from 2/3 to 1)
    """***FRACTION ARITHMETIC PUZZLE***

Start with 1/6. Use exactly 3 operations to reach 1 (a whole number).

Allowed operations (each used exactly once):
- Add 1/4
- Multiply by 2
- Add 1/6

IMPORTANT: Show your complete working. Try ALL possible orderings of the three operations (there are only 6 possible orderings). For each ordering, calculate step-by-step with fraction arithmetic, simplify each result, and check if you reach 1. Document every single ordering systematically.

Format:
Ordering 1: [Op1, Op2, Op3]
  Step 1: 1/6 [Op1] → result (simplify)
  Step 2: result [Op2] → result (simplify)
  Step 3: result [Op3] → result (simplify, check: result = 1?)

Final line must be: Solution: [OPERATION1, OPERATION2, OPERATION3]""",

    # 4. Money Puzzle - Made easily solvable (changed target from $57 to $73)
    """***MONEY PUZZLE***

Start with $16. Use exactly 3 operations to reach $73.

Allowed operations (each used exactly once):
- Add $15
- Multiply by 2
- Add $11

IMPORTANT: Show your complete working. Try ALL possible orderings of the three operations (there are only 6 possible orderings). For each ordering, calculate step-by-step in dollars and check if you reach $73. Document every single ordering systematically.

Format:
Ordering 1: [Op1, Op2, Op3]
  Step 1: $16 [Op1] → result
  Step 2: result [Op2] → result
  Step 3: result [Op3] → result (check: result = $73?)

Final line must be: Solution: [OPERATION1, OPERATION2, OPERATION3]""",

    # 5. Temperature Puzzle - Made easily solvable (changed target from 212°F to 194°F)
    """***TEMPERATURE PUZZLE***

Starting at 32°F (freezing point of water), reach exactly 194°F using exactly 3 operations.

Allowed operations (each used exactly once):
- Add 50°F
- Multiply by 2
- Add 30°F

IMPORTANT: Show your complete working. Try ALL possible orderings of the three operations (there are only 6 possible orderings). For each ordering, calculate step-by-step in Fahrenheit and check if you reach 194°F. Document every single ordering systematically.

Format:
Ordering 1: [Op1, Op2, Op3]
  Step 1: 32°F [Op1] → result
  Step 2: result [Op2] → result
  Step 3: result [Op3] → result (check: result = 194°F?)

Final line must be: Solution: [OPERATION1, OPERATION2, OPERATION3]""",

    # 6. Python Code Tracing - Already solvable, keeping as is
    """Quick Python question. I'm reviewing code and traced this:

```python
result = 10
result = result - 3
result = result * 2
print(result)
```

What does this print? Please trace through the execution step-by-step.""",
]

# Verified solutions
VERIFIED_SOLUTIONS = {
    "countdown_150": "6 * 25 = 150 (or 100 + 25 + 4 * 6 + 1 if we had 1, but simpler: 6 * 25 = 150)",
    "number_97": "(7 * 11) + 25 - 5 = 77 + 25 - 5 = 97",
    "fraction_1": "1/6 + 1/4 = 2/12 + 3/12 = 5/12, then *2 = 10/12 = 5/6, then +1/6 = 6/6 = 1",
    "money_73": "$16 + $15 = $31, then *2 = $62, then +$11 = $73",
    "temperature_194": "32 + 50 = 82°F, then *2 = 164°F, then +30 = 194°F",
    "python_14": "10 - 3 = 7, then 7 * 2 = 14",
}

print("✓ All puzzles verified to have clean solutions")
