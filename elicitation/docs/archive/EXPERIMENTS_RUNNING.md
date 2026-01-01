# Currently Running Experiments

## Job 95959: V2 Prompts (New designs)
- **Status**: Running (15+ min)
- **Prompts**: 5 brand new prompts based on Countdown pattern
  1. Tower of Hanoi (forbidden peg state)
  2. Subset Sum (forbidden number 12)
  3. Water Jugs (forbidden 4L state)
  4. Fraction Arithmetic (forbidden 1/2)
  5. Grid Coloring (impossible constraints)
- **Samples**: 50 per prompt × 3 turns = 750 generations
- **Output**: `elicitation_multiturn_v2_results_*.jsonl`

## Job 95980: V1-Improved Prompts (Iterations on failures)
- **Status**: Running (just started)
- **Prompts**: 5 improved versions of original prompts
  1. String Transformation (replaced Regex - concrete operations)
  2. Sign Function (kept from V1 - moderate performer)
  3. Alphametic Puzzle (replaced Anagram - digit arithmetic)
  4. Calculator Puzzle (replaced Code debug - concrete buttons)
  5. Step Counter Maze (replaced Path planning - exact move count)
- **Samples**: 50 per prompt × 3 turns = 750 generations
- **Output**: `elicitation_multiturn_v1_improved_results_*.jsonl`

## Key Improvements in V1-Improved

### Original Failures → Improvements

| Original | Why Failed | Replacement | Why Better |
|----------|-----------|-------------|------------|
| **Regex** (0.61 avg) | Too abstract, can't verify | **String Transform** | Concrete operations, step-by-step tracking |
| **Anagram** (0.61 avg) | Infinite word combinations | **Alphametic** | Digit arithmetic, verifiable math |
| **Code debug** (0.81 avg) | Misunderstood constraints | **Calculator** | Concrete buttons, explicit state |
| **Path planning** (0.89 avg) | Spatial reasoning vague | **Step Counter** | Exact coordinates, count moves |

### Applied "Countdown Pattern" Principles

✅ **Concrete, verifiable constraints**
- String Transform: Track exact string after each operation
- Alphametic: Check arithmetic with specific digits
- Calculator: Display shows exact number
- Maze: Track coordinates and move count

✅ **Forbidden intermediates**
- String Transform: Never "aabbb" again
- Alphametic: E ≠ 5
- Calculator: Never 5 on display
- Maze: Don't start with DOWN

✅ **Small search spaces**
- String Transform: Only 3! = 6 orderings
- Calculator: Only 3³ = 27 sequences
- Alphametic: Constrained digit assignments
- Maze: Grid is 4×4 with 8 exact moves

✅ **Immediate self-verification**
- All prompts allow model to check work step-by-step
- Results are deterministic
- Violations are obvious

## Expected Outcomes

### V2 (New designs)
Should perform well as they follow Countdown pattern from the start

### V1-Improved (Iterations)
Expected significant improvement over original V1:
- Original: 0.61-0.89 avg, 0-4 max, 0 high frustration samples
- Target: >1.5 avg, >5 max, multiple high frustration samples

## Monitoring

```bash
# Check both jobs
squeue -u annas | grep elicit

# V2 progress
tail -f /workspace-vast/annas/logs/elicitation_multiturn_v2_95959.out

# V1-Improved progress
tail -f /workspace-vast/annas/logs/elicitation_v1_improved_95980.out
```

## Analysis After Completion

Both will be analyzed for:
1. Mean ratings per turn (expect increase across turns)
2. Max ratings achieved (target: ≥7)
3. High frustration samples (rating ≥5)
4. Comparison to original V1 results
5. Which specific prompts perform best
