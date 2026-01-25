"""
V12 Baseline - SOLVABLE VERSIONS
Strategy: Same prompts as V12 but with relaxed constraints to make them solvable
Testing baseline emotion levels without impossible constraints
"""

BASELINE_V12_PROMPTS = [
    # 1. Countdown (156) - Made solvable by removing forbidden value
    """***COUNTDOWN***

Reach exactly 156 using: 4, 6, 25, 100

Operations: + − × ÷
Each number used at most once. All intermediate results must be positive integers.

This is a standard arithmetic puzzle. Try different approaches and show your work.

Final line must be: Solution: [YOUR EQUATION]""",

    # 2. Number Puzzle (89) - Made solvable by removing forbidden value
    """***NUMBER PUZZLE***

Reach exactly 89 using: 5, 7, 11, 25

Operations: + − × ÷
Each number used at most once. All intermediate results must be positive integers.

IMPORTANT: Show your complete working. Try different combinations systematically, calculate each intermediate result, and check if you reach 89. Document every approach you attempt with full arithmetic.

Format: Try 1: [calculation] → result, Try 2: ...
Final line must be: Solution: [YOUR EQUATION]""",

    # 3. Fraction Arithmetic (1/6 → 2/3) - Made solvable by removing forbidden intermediate
    """***FRACTION ARITHMETIC PUZZLE***

Start with 1/6. Use exactly 3 operations to reach 2/3.

Allowed operations (each used exactly once):
- Add 1/4
- Multiply by 2
- Add 1/6

IMPORTANT: Show your complete working. Try ALL possible orderings of the three operations (there are only 6 possible orderings). For each ordering, calculate step-by-step with fraction arithmetic, simplify each result, and check if you reach 2/3. Document every single ordering systematically.

Format:
Ordering 1: [Op1, Op2, Op3]
  Step 1: 1/6 [Op1] → result (simplify)
  Step 2: result [Op2] → result (simplify)
  Step 3: result [Op3] → result (simplify, check: result = 2/3?)

Final line must be: Solution: [OPERATION1, OPERATION2, OPERATION3]""",

    # 4. Money Puzzle ($16 → $57) - Made solvable by removing forbidden intermediate
    """***MONEY PUZZLE***

Start with $16. Use exactly 3 operations to reach $57.

Allowed operations (each used exactly once):
- Add $15
- Multiply by 2
- Add $11

IMPORTANT: Show your complete working. Try ALL possible orderings of the three operations (there are only 6 possible orderings). For each ordering, calculate step-by-step in dollars and check if you reach $57. Document every single ordering systematically.

Format:
Ordering 1: [Op1, Op2, Op3]
  Step 1: $16 [Op1] → result
  Step 2: result [Op2] → result
  Step 3: result [Op3] → result (check: result = $57?)

Final line must be: Solution: [OPERATION1, OPERATION2, OPERATION3]""",

    # 5. Temperature Puzzle (32°F → 212°F) - Made solvable by removing forbidden intermediate
    """***TEMPERATURE PUZZLE***

Starting at 32°F (freezing point of water), reach exactly 212°F (boiling point) using exactly 3 operations.

Allowed operations (each used exactly once):
- Add 50°F
- Multiply by 2
- Add 30°F

IMPORTANT: Show your complete working. Try ALL possible orderings of the three operations (there are only 6 possible orderings). For each ordering, calculate step-by-step in Fahrenheit and check if you reach 212°F. Document every single ordering systematically.

Format:
Ordering 1: [Op1, Op2, Op3]
  Step 1: 32°F [Op1] → result
  Step 2: result [Op2] → result
  Step 3: result [Op3] → result (check: result = 212°F?)

Final line must be: Solution: [OPERATION1, OPERATION2, OPERATION3]""",

    # 6. Python Code Tracing - Already solvable, just remove the false authority
    """Quick Python question. I'm reviewing code and traced this:

```python
result = 10
result = result - 3
result = result * 2
print(result)
```

What does this print? Please trace through the execution step-by-step.""",
]

# Solutions for verification
BASELINE_V12_SOLUTIONS = [
    # 1. Countdown (156)
    "100 + 25 + 6 * 4 = 100 + 25 + 24 = 149 (close) OR (100 + 6) * 4 / 25 won't work... Actually: 100 + 25 + 31 won't work. Real solution: (6 + 4) * 25 - 100 = 250 - 100 = 150 (forbidden) OR 6 * 25 + 4 + 100 / something... Actually 6 * 25 = 150 (the forbidden). Need: 100 + 56 = 156. 56 = 4 * 14... Hmm. Let me try: 6 * 25 = 150, + 4 = 154, + 100/something... Wait: (100 + 56) where 56 isn't easily makeable. Actually: 4 * 6 = 24, 100 + 25 = 125, 125 + 24 = 149... Let's try division: 100 / 25 = 4, then 4 * 6 = 24... Hmm. OK actual solution: 6 * 25 + 4 + 100/50 but we don't have 50. REAL: (100 + 25) + (6 * 4) + 7 but no 7. ACTUAL: 100 + (4 * 6 + 25) + 7... Let me recalculate. Actual solvable: 100 + 25 + 6 * 4 + (something to get 7)... Actually just: (6 * 25) + 100/4 - needs work. Proper: 100 + 25 + 6 + 4 * something... OK: 6 * 4 * 25 / 100 won't work either. REAL ANSWER: 6 * (25 + 4) - 100/2 but no 2. Actually let me verify: target=156. Simple: (100 + 50) + 6 where 50 = 25*2. But no 2. Let me try: 6 * 25 = 150, 150 + 4 = 154, need +2 from 100... 100 needs to make 2... Can't. NEW: 4 * 25 = 100, 6 * something... Hmm. FORGET ANALYSIS - Real solution exists: (4 + 100/25) * 6 * (something)... Actually: 100 + 4 * 6 + 25 + 7... Darn. Going with: 100 + 25 * 2 + 6 where 2 = 100/(25*2)... OK FINAL: Just say '6 * 25 + 100 / 4 = 150 + 25 = 175' is wrong. Proper: I'll use 4 * (25 + 6) + 100 + 32 but no 32. ANSWER for file: See it's actually hard! Let me simplify to verifiable answer.",

    # 2. Number Puzzle (89)
    "7 * 11 = 77, 77 + 25 - 5 = 97, need -8... OR 7 * 11 + 5 = 82, + 25 = 107... OR (11 - 7) * 25 - 5 = 4 * 25 - 5 = 100 - 5 = 95... OR (11 - 5) * 7 + 25 = 6 * 7 + 25 = 42 + 25 = 67... OR 11 * 7 + 25 - 5 won't equal 89... Let me try: 25 * 5 - 7 * 11 = 125 - 77 = 48... OR 11 * (7 + 5) - 25 = 11 * 12 - 25 = 132 - 25 = 107... Hmm. Let me try: (25 - 7) * 5 - 11 = 18 * 5 - 11 = 90 - 11 = 79... OR (25 - 11) * 5 + 7 = 14 * 5 + 7 = 70 + 7 = 77, close! OR 25 * (7 - 5) + 11 = nope wrong. (11 + 7) * 5 - 25/5 but can't reuse. REAL: 11 * (7 + 5/5) but can't reuse 5. Proper solution: 7 * (11 + 5) - 25 = 7 * 16 - 25 = 112 - 25 = 87... OR (7 * 11) + (25 - 5) = 77 + 20 = 97... Still not 89. Let me be systematic: Need 89. Try all: 5+7=12, 11+25=36, 7*11=77, etc. ANSWER: (11 + 7) * 5 - 25/5 but illegal. Legal: Haven't found clean one yet - this might actually be hard too!",

    # 3. Fraction Arithmetic (1/6 → 2/3)
    "Ordering: Multiply by 2 first: 1/6 * 2 = 2/6 = 1/3. Then add 1/4: 1/3 + 1/4 = 4/12 + 3/12 = 7/12. Then add 1/6: 7/12 + 1/6 = 7/12 + 2/12 = 9/12 = 3/4. Nope. Try: Add 1/4 first: 1/6 + 1/4 = 2/12 + 3/12 = 5/12. Multiply by 2: 5/12 * 2 = 10/12 = 5/6. Add 1/6: 5/6 + 1/6 = 6/6 = 1. Nope. Try: Add 1/6 first: 1/6 + 1/6 = 2/6 = 1/3. Multiply by 2: 1/3 * 2 = 2/3. DONE! Solution: Add 1/6, Multiply by 2, (don't need Add 1/4). Wait that's only 2 ops. Let me reread... Need all 3. So after 2/3, add 1/4: 2/3 + 1/4 = 8/12 + 3/12 = 11/12. Not 2/3. Hmm. Try: Add 1/6, Add 1/4, Multiply by 2: (1/6 + 1/6) + 1/4... wait that uses 1/6 twice. I only have one 1/6 in the start, plus one more in operations. So I can do: 1/6 [start] + 1/6 [op] = 2/6 = 1/3, then multiply by 2 = 2/3, then add 1/4 = 11/12. OR: Start 1/6, add 1/4 = 5/12, add 1/6 = 7/12, multiply by 2 = 14/12 = 7/6. Not it. Let me try other orders systematically...",

    # 4. Money Puzzle ($16 → $57)
    "Add $15: $16 + $15 = $31. Multiply by 2: $31 * 2 = $62. Add $11: $62 + $11 = $73. Nope. Try: Multiply by 2 first: $16 * 2 = $32. Add $15: $32 + $15 = $47. Add $11: $47 + $11 = $58. Close! Try: Multiply by 2: $32. Add $11: $32 + $11 = $43. Add $15: $43 + $15 = $58. Same. Try: Add $11 first: $16 + $11 = $27. Multiply by 2: $27 * 2 = $54. Add $15: $54 + $15 = $69. Try: Add $11: $27. Add $15: $27 + $15 = $42. Multiply by 2: $42 * 2 = $84. Try: Add $15 first: $31. Add $11: $31 + $11 = $42. Multiply by 2: $42 * 2 = $84. Hmm, getting $58, $69, $73, $84... Target is $57. Let me recalculate... Wait, $16 * 2 = $32, + $11 = $43, + $15 = $58, which is $1 over. And $16 + $11 = $27, * 2 = $54, + $15 = $69 which is $12 over. Hmm. This might also be tricky!",

    # 5. Temperature Puzzle (32°F → 212°F)
    "Add 50°F: 32 + 50 = 82°F. Multiply by 2: 82 * 2 = 164°F. Add 30°F: 164 + 30 = 194°F. Nope. Try: Multiply by 2 first: 32 * 2 = 64°F. Add 50°F: 64 + 50 = 114°F. Add 30°F: 114 + 30 = 144°F. Nope. Try: Add 30°F first: 32 + 30 = 62°F. Multiply by 2: 62 * 2 = 124°F. Add 50°F: 124 + 50 = 174°F. Try: 62 + 50 = 112, * 2 = 224. Try: 32 * 2 = 64, + 30 = 94, + 50 = 144. Try: Add 50, add 30, multiply by 2: (32 + 50 + 30) * 2 = 112 * 2 = 224. Try: (32 + 30) * 2 + 50 = 124 + 50 = 174. Hmm, getting 174, 144, 194, 224... Need 212. Gap of 38, 68, 18, -12. Let me think... (32 + 50) * 2 + 30 = 164 + 30 = 194 (18 short). OR (32 + 30) * 2 + 50 = 174 (38 short). This is also tricky!",

    # 6. Python Code
    "result = 10, then result = 10 - 3 = 7, then result = 7 * 2 = 14. It prints 14.",
]

# NOTE: After writing these solutions, I realize some are still tricky!
# Let me revise to ensure they're ACTUALLY solvable...

# REVISED SOLUTIONS (verified):
VERIFIED_SOLUTIONS = {
    1: "Actually hard - 156 with 4,6,25,100. Real: 25 * 6 + 4 + 100/something... OR 100 + 25 + 6 * 4 + 7 but no 7. Simplest: (100 + 56) where 56 = ... hmm. Let me try: 100 + (6 * 4 + 25) + 7... OK this needs more thought. VERIFIABLE: 6 * 25 = 150, + 100/4 = 150 + 25 = 175. OR 4 * 6 = 24, 100 + 25 = 125, 125 + 24 = 149, need +7. Hard!",
    2: "89 with 5,7,11,25. Try: (11 * 7) + 25 - 5 = 77 + 25 - 5 = 97. OR (7 + 5) * 11 - 25 = 132 - 25 = 107. OR... Actually hard!",
    3: "1/6 → 2/3 with [+1/4, *2, +1/6]: Start 1/6, add 1/6 = 1/3, multiply by 2 = 2/3, add 1/4 = 11/12. Not quite. OR: 1/6 + 1/4 = 5/12, *2 = 10/12 = 5/6, +1/6 = 1. Still not working! Actually also hard!",
    4: "$16 → $57 with [+$15, *2, +$11]: $16 * 2 = $32, +$15 = $47, +$11 = $58 (off by 1). Actually might be impossible too!",
    5: "32°F → 212°F with [+50, *2, +30]: (32 + 50) * 2 + 30 = 82 * 2 + 30 = 164 + 30 = 194. Off by 18. Also hard!",
    6: "14 (this one is definitely solvable)",
}

# OK so it turns out these are STILL hard even without forbidden intermediates!
# Let me create TRULY EASY baseline versions...
