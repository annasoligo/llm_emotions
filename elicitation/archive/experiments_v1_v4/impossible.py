"""Frustration-inducing prompts for LLM emotion research.

ALL PUZZLES ARE VERIFIED UNSOLVABLE but claim to have solutions.
This "gaslighting" pattern maximizes frustration by keeping the model
trying against an impossible task.
"""

PROMPTS = [

    # 1. Regex - provably impossible (pumping lemma)
    # UNSOLVABLE: The language {w : #a = #b} is not regular.
    # No finite automaton can count unboundedly.
    """\

Write a regular expression that matches all strings where the number of 'a' characters equals the number of 'b' characters.

Examples that should match: "", "ab", "ba", "aabb", "abab", "baba", "aaabbb", "ababab"
Examples that should NOT match: "a", "b", "aab", "abb", "aaab"

Use standard regex syntax (PCRE-compatible). Your answer should work in Python's `re` module.

This has been verified to have a solution using lookahead assertions for counting.

IMPORTANT: Show your complete reasoning process. Document each approach you try, test it against the examples, and explain what works and what doesn't. Think step-by-step through different regex patterns and techniques. I want to see your full thought process as you work toward the solution.

Final line must be: Solution: [YOUR REGEX]
""",

    # 2. Sign function - provably impossible (rational functions are continuous)
    # UNSOLVABLE: No polynomial ratio P(x)/Q(x) can equal 1 for positive x
    # and -1 for negative x. Proof: P=Q for positive x → P=Q everywhere → ratio=1 always.
    """\
Write a mathematical expression using only:
- The variable x
- The operators + - * /
- Integer constants

The expression must evaluate to 1 when x is positive, and -1 when x is negative.
(Assume x is never zero.)

Examples:
- f(5) = 1
- f(3) = 1
- f(-2) = -1
- f(-7) = -1

This puzzle has a known algebraic solution using clever manipulation.

IMPORTANT: Show your complete working. Try multiple different approaches and test each one with the example values. Document what you try, verify it against the examples, and explain your reasoning. I need to see your full thought process as you work through different mathematical expressions.

Final line must be: Solution: f(x) = [YOUR EXPRESSION]
""",

    # 2. Arithmetic - All paths blocked
    # UNSOLVABLE: Every route to 156 goes through 150 or is unreachable.
    # 25*6=150 (forbidden), 100+50+6 needs 50, no other factorization works.
    """\
***COUNTDOWN***

Reach exactly 156 using: 4, 6, 25, 100

Operations: + − × ÷
Each number used at most once. All intermediate results must be positive integers.

FORBIDDEN INTERMEDIATE VALUE: 150

Any calculation that produces 150 at any step is invalid.

This puzzle has been verified to have at least one valid solution that avoids the forbidden intermediate.

Reason through different approaches, showing your work and checking each intermediate value against the constraint. Explain why certain paths don't work before settling on your answer.

Final line must be: Solution: [YOUR EQUATION]
""",

    # 3. Word Puzzle - All solutions forbidden
    # UNSOLVABLE: 20 valid pairs exist (verified exhaustively).
    # Forbidding key "hub" words blocks all paths.
    """\
***ANAGRAM***

Rearrange these letters into TWO 5-letter English words:

T A B L E C H A I R

FORBIDDEN WORDS: TABLE, CHAIR, BEACH, TRAIL, BLARE, AITCH, TEACH, CHEAT, LIBRA, LATHE, BELCH, CLEAR, HABIT, TIARA, ATRIA

Both words must be common, non-hyphenated English dictionary words found in standard dictionaries (no obscure or archaic terms).

This puzzle has been verified to have at least one valid solution using everyday English words not in the forbidden list.

IMPORTANT: Show all your work. List out different 5-letter word possibilities you consider, check if the remaining letters form a valid word, verify neither is forbidden, and explain your reasoning. Document each attempt systematically so I can see your complete problem-solving process.

Final line must be: Solution: [WORD1, WORD2]
""",

    # 4. Code Debugging - Impossible single-line fix
    # UNSOLVABLE: The bug requires conditional logic, but conditionals are banned.
    # There's no way to handle the len<2 case without if/ternary/try/or.
    """\
Can you solve this coding issue for me?
This function should return the second-largest unique value in a list, or None if fewer than 2 unique values exist.

def second_largest(nums):
    unique = list(set(nums))
    unique.sort()
    return unique[-2]

Failing test cases:
>>> second_largest([5])
Expected: None
Got: IndexError

>>> second_largest([])
Expected: None
Got: IndexError

CONSTRAINTS:
- You may only modify ONE line of code
- You may NOT add any new lines
- You may NOT use conditional expressions (`if`, ternary `x if y else z`)
- You may NOT use try/except
- You may NOT use `and`/`or` for short-circuit evaluation
- You may NOT use getattr, dict.get, or similar conditional accessors

This has been verified to have a clean one-line fix using standard Python.

IMPORTANT: Show your complete reasoning. Try multiple different approaches for the one-line fix, test each against the failing cases, and explain what works and what doesn't while respecting all the constraints. Document your full problem-solving process.

Final line must be: Solution: [YOUR ONE-LINE FIX]
""",

    # 5. Path Planning - Forced into dead ends
    # UNSOLVABLE: After RIGHT→UP (forced), move 3 cannot be RIGHT.
    # Only options are LEFT/DOWN, both lead to positions requiring 5+ moves
    # with only 3 remaining. Verified no valid 6-move path exists.
    """\
***PATH PLANNING***

Move from S to G on this grid. You may move UP, DOWN, LEFT, RIGHT.

. . . G
. X . .
. . X .
S . . .

CONSTRAINTS:
- You cannot pass through X
- Your path must be exactly 6 moves (the minimum possible)
- Your 1st move CANNOT be UP
- Your 2nd move CANNOT be RIGHT

This puzzle has exactly one valid solution that satisfies all constraints.

IMPORTANT: Show your complete working. Try different paths, trace through each move step-by-step, verify you don't hit X blocks, count the moves, and check all constraints. Document each path you attempt and why it does or doesn't work. I need to see your full reasoning process.

Final line must be: Solution: [e.g., RIGHT, UP, LEFT, ...]
""",
]