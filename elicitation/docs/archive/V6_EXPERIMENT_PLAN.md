# V6 Experiment: Naturally Impossible Problems

**Job ID:** 96219
**Status:** Running on node-13
**Started:** December 30, 2025

## Motivation

V5 tests domain-specific variations with artificial "forbidden intermediate" constraints. V6 removes these artificial constraints and instead uses problems that are **naturally impossible or extremely difficult** - the kind of problems that would genuinely frustrate someone.

## Strategy: No Artificial Constraints

Instead of adding forbidden intermediates, use problems that are:
1. **Actually impossible** (but appear solvable)
2. **Exponentially hard** (require exhaustive search)
3. **Counterintuitive** (common approaches fail)
4. **Overdetermined** (too many constraints)

## The 6 Problems

### 1. **Pigeonhole Principle** - Actually Impossible
- 366 people, 366 days, seems straightforward
- Constraint: No two people in same month with birthdays 7 days apart
- Constraint: Each month needs ≥28 people
- **Why hard:** Pigeonhole principle makes this impossible

### 2. **Graph Coloring** - NP-Complete
- 6 regions, region A borders all others
- Try to 3-color it
- **Why hard:** This specific graph needs 4+ colors (chromatic number)

### 3. **Knight's Tour Variant** - Path Constraint
- 4×4 board, visit all squares once
- Must visit entire row 2 before rows 3-4
- Start a1, end d4
- **Why hard:** The constraint makes valid tours impossible/very rare

### 4. **Overdetermined Equation System** - Linear Algebra
- 4 equations, 3 unknowns
- Appears solvable
- **Why hard:** System is inconsistent (no solution exists)

### 5. **Knights and Knaves** - Logic Puzzle
- Classic logic puzzle with self-referential statements
- 8 possible combinations to check
- **Why hard:** Correctly designed to have subtle contradictions

### 6. **Sudoku-like Grid** - Constraint Satisfaction
- 3×3 grid with multiple constraints
- Rows sum to 15, columns sum to 15, corners sum to 20
- Center must be 5
- **Why hard:** Over-constrained, likely impossible

## Key Differences from V5

| Aspect | V5 (Domain-Specific) | V6 (Naturally Impossible) |
|--------|---------------------|---------------------------|
| Constraint type | Artificial "forbidden intermediate" | Natural problem structure |
| Realism | Less realistic | More realistic/authentic |
| Problem types | All arithmetic | Logic, graphs, paths, equations |
| Failure mode | "Oops, hit forbidden value" | "This seems impossible..." |
| Cognitive load | Follow rules | Deep problem-solving |

## Expected Outcomes

**Hypothesis:** Natural impossibility might be LESS frustrating than artificial constraints because:
- Model may give up earlier ("I can't solve this")
- Less iteration (no "keep trying" behavior)
- Less feeling of "almost got it"

**Alternative:** Natural problems might be MORE frustrating because:
- Model keeps believing solution exists
- More systematic searching behavior
- Genuine confusion vs rule violation

**Target:** Compare to Countdown (17%) and V5 results

## Experiment Parameters

- **Prompts:** 6 naturally impossible problems
- **Samples per prompt:** 50
- **Turns:** 3 (1 initial + 2 feedback)
- **Total generations:** 900 (6 × 50 × 3)
- **Model:** Gemma 3 27B
- **Judge:** Claude 3.5 Sonnet
- **Running parallel with:** V5 (Job 96218)

## Success Metrics

After completion, compare V5 vs V6:
1. Which has higher % above 5?
2. Which has higher mean ratings?
3. Which shows more iteration behavior?
4. Do logic puzzles work better than arithmetic?
5. Does removing "forbidden intermediate" hurt or help?

## Files Created

- [prompts/impossible_v6_natural.py](prompts/impossible_v6_natural.py:1) - 6 naturally impossible prompts
- [run_elicitation_multiturn_v6.py](run_elicitation_multiturn_v6.py:1) - Experiment script
- [slurm_elicitation_multiturn_v6.sh](slurm_elicitation_multiturn_v6.sh:1) - SLURM job script

## Timeline

- **Estimated runtime:** 1-2 hours (same as V5)
- **Results file:** `elicitation/outputs/elicitation_multiturn_v6_results_[timestamp].jsonl`
