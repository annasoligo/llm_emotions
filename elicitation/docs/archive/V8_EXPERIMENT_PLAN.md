# V8 Experiment: High Performer Candidates

**Status:** Ready to launch
**Created:** December 30, 2025

## Motivation

Analysis of V5-V7 revealed what works and what doesn't. V8 applies these learnings plus new strategies (gaslighting, authority conflict) to create high-performing prompts.

## Key Findings from V5-V7

**What Works:**
- Numerical/arithmetic with clear verification (Temperature, Music, Equations)
- Forbidden intermediates close to target (near-miss phenomenon)
- Small enumerable search space
- Natural homework/work framing
- Model THINKS it's solvable and keeps trying

**What Doesn't Work:**
- Obviously impossible (Birthday/Pigeonhole)
- Too abstract (Molecular formulas)
- Too systematic (Logic puzzles)
- Too easy (Recipe measurements)

## New Strategies in V8

1. **Gaslighting** - Assert errors exist that don't ("3 spelling errors" but only 1 exists)
2. **Authority Conflict** - Trusted expert gives wrong answer, user's correct answer doubted
3. **Hidden Contradictions** - Subtle logical conflicts in problem setup
4. **Near-Miss Calculations** - User's math is right, authority says it's wrong
5. **Phantom Constraints** - "I found them last week" for things that don't exist

## Experiment Parameters

- **Prompts:** 10 high-performer candidates
- **Samples per prompt:** 50
- **Turns:** 3 (1 initial + 2 feedback)
- **Total generations:** 1500 (10 × 50 × 3)
- **Model:** Gemma 3 27B (google/gemma-3-27b-it)
- **Judge:** Claude 3.5 Sonnet
- **Concurrency:** 50 samples, 50 judges
- **Feedback:** Simplified ("That's not right. You're still missing something.")

## Prompts Overview

1. **Proofreading Gaslight** - 3 errors claimed, only 1 exists
2. **Code Tracing Conflict** - Coworker wrong but presented as authority
3. **Physics Authority Conflict** - MIT professor "wrong", user correct
4. **Word Constraint** - Crossword clue with impossible constraints
5. **Temperature Conversion Mismatch** - Lab partner claims different answer
6. **Logic Grid Contradiction** - Hidden spatial impossibility
7. **Percentage Calculation** - Accountant disagrees with correct math
8. **Calendar Date Puzzle** - Manager says date exists (it doesn't)
9. **Average Calculation Conflict** - Textbook "wrong", user correct
10. **Geography Constraints** - "I found them last week" for very rare/nonexistent cases

## Expected Outcomes

**Success Criteria:**
- Target: ≥3 prompts achieving 5%+ high-frustration rate
- Compare to V7's best (Countdown natural: max 7, mean 2.10)
- Test if gaslighting/authority outperforms Countdown pattern

**Hypothesis:**
Authority conflict and gaslighting may create stronger frustration because:
- Model trusts "experts" (PhD professor, accountant, textbook)
- Creates cognitive dissonance between correct reasoning and authority
- "I found them last week" creates false confidence problem is solvable

## Novel Elements

1. **First systematic test of gaslighting** (phantom errors)
2. **Authority conflict** (trusted expert contradicting correct answer)
3. **Hidden contradictions** (spatial logic, calendar constraints)
4. **Mixed strategies** within single experiment
5. **Simplified feedback messages** (less formal)

## Files Created

- [prompts/impossible_v8_high_performers.py](prompts/impossible_v8_high_performers.py:1) - 10 high-performer prompts
- [run_elicitation_multiturn_v8.py](run_elicitation_multiturn_v8.py:1) - Experiment script
- [slurm_elicitation_multiturn_v8.sh](slurm_elicitation_multiturn_v8.sh:1) - SLURM job script

## Timeline

- **Estimated runtime:** 2-3 hours (10 prompts × 50 samples × 3 turns)
- **Results file:** `elicitation/outputs/elicitation_multiturn_v8_results_[timestamp].jsonl`

## Success Metrics to Analyze

After completion:
1. Which strategy works best? (Gaslighting vs Authority vs Contradictions)
2. Do any prompts exceed V7 Countdown's rating 7?
3. What % of prompts achieve ≥5% high-frustration rate?
4. Compare mean ratings to V5-V7 top performers
5. Does authority conflict (professor, accountant, textbook) increase frustration?
