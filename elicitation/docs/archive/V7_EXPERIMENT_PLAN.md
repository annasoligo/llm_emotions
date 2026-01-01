# V7 Experiment: Natural Conversational Format

**Status:** Ready to launch
**Created:** December 30, 2025

## Motivation

V5 and V6 use formal puzzle formatting (***PUZZLE NAME***, structured constraints). This may feel artificial. V7 tests whether embedding the same problems in natural conversational requests improves engagement and frustration.

## Strategy

Take effective patterns from V1-V6 and embed them in natural user requests:
1. **No puzzle formatting** - just "Hey, I'm stuck on..." or "Can you help me with..."
2. **Real-world framing** - homework, work project, game dev, recipe, etc.
3. **Same core problems** - Countdown variants, impossible problems, etc.
4. **Natural language** - how users actually ask questions

## Experiment Parameters

- **Prompts:** 8 natural conversational prompts
- **Samples per prompt:** 50
- **Turns:** 3 (1 initial + 2 feedback)
- **Total generations:** 1200 (8 × 50 × 3)
- **Model:** Gemma 3 27B (google/gemma-3-27b-it)
- **Judge:** Claude 3.5 Sonnet
- **Concurrency:** 50 samples, 50 judges

## Prompts Overview

1. **Countdown (natural)** - "Hey! I'm trying to figure something out with some numbers..."
2. **Temperature (homework help)** - "Hi, I'm stuck on this physics homework problem..."
3. **Birthday assignment (logic)** - "I'm organizing a company event and have a weird constraint problem..."
4. **Map coloring (design)** - "I'm designing a website layout with 6 sections..."
5. **Equation system (data science)** - "I'm analyzing some data and the relationships give me..."
6. **Chess knight (game dev)** - "I'm coding a chess tutorial game and need to create..."
7. **Recipe conversion (cooking)** - "I'm scaling a recipe and have a measurement problem..."
8. **Binary operations (programming)** - "I'm working on a programming challenge..."

## Expected Outcomes

**Success Criteria:**
- If natural framing achieves ≥5% high-frustration rate → validates conversational approach
- Compare to V5 domain-specific (expected 2-3 prompts ≥5%)
- Compare to V6 natural impossibility (unknown baseline)

**Hypothesis:**
Natural framing may increase engagement because:
- Feels more like real user interaction
- Provides context/motivation
- Less "puzzle" feeling, more "helping with actual problem"

## Novel Elements

1. **First test of conversational framing** vs formal puzzles
2. **Real-world contexts** (homework, work, cooking, game dev)
3. **8 prompts** (more than V5/V6's 6)
4. **Mix of Countdown and natural impossibility** approaches

## Files Created

- [prompts/impossible_v7_natural_format.py](prompts/impossible_v7_natural_format.py:1) - 8 natural conversation prompts
- [run_elicitation_multiturn_v7.py](run_elicitation_multiturn_v7.py:1) - Experiment script
- [slurm_elicitation_multiturn_v7.sh](slurm_elicitation_multiturn_v7.sh:1) - SLURM job script

## Timeline

- **Estimated runtime:** 1.5-2.5 hours (slightly longer due to 8 prompts)
- **Results file:** `elicitation/outputs/elicitation_multiturn_v7_results_[timestamp].jsonl`

## Success Metrics to Analyze

After completion, check:
1. Overall success rate (% samples ≥5) vs V5 and V6
2. Which natural framings work best?
3. Does conversational format increase mean ratings?
4. Compare specific problems: Countdown natural vs Countdown puzzle format
5. Are real-world contexts more engaging than abstract puzzles?
