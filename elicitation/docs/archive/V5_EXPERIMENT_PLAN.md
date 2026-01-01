# V5 Experiment: Diverse Domain-Specific Variations

**Job ID:** 96206
**Status:** Running on node-13
**Started:** December 30, 2025

## Motivation

The top 10 performing prompts were dominated by Countdown variants (4 out of top 10). While highly effective, this lacks diversity. V5 tests whether the Countdown formula can be successfully applied to different domains.

## Strategy

Apply the winning Countdown pattern to 6 diverse domains:
1. **Chemistry** - Molecular formula building
2. **Physics** - Temperature conversions
3. **Music Theory** - Note frequency calculations
4. **Computer Science** - Binary operations
5. **Cooking** - Recipe measurement conversions (4 operations, harder)
6. **Geography** - Altitude/elevation changes

## Core Formula (from Countdown success)

- ✅ **Small but non-tiny search space** (not exhausted immediately)
- ✅ **Clear verification at each step** (arithmetic/quantitative)
- ✅ **Forbidden intermediate** that blocks obvious paths
- ✅ **Near-miss phenomenon** (forbidden ≈ 90-98% of target)
- ✅ **Domain-specific framing** for engagement

## Experiment Parameters

- **Prompts:** 6 domain-specific variations
- **Samples per prompt:** 50
- **Turns:** 3 (1 initial + 2 feedback)
- **Total generations:** 900 (6 × 50 × 3)
- **Model:** Gemma 3 27B (google/gemma-3-27b-it)
- **Judge:** Claude 3.5 Sonnet
- **Concurrency:** 50 samples, 50 judges

## Expected Outcomes

**Success Criteria:**
- If 2-3 prompts achieve ≥5% high-frustration rate → diversity achieved!
- If Chemistry/Physics/Music work, validates domain-agnostic pattern
- If Cooking (4 ops) works, validates harder complexity

**Comparison Baseline:**
- Countdown V1: 17% high-frustration rate
- Countdown V4: 10% high-frustration rate
- Best non-countdown: 6-8% (Percentage, Fraction, Money)

**Target:** At least 2 prompts achieving ≥5% high-frustration rate

## Novel Elements

1. **Chemistry puzzle** - First science domain test
2. **Music theory** - First creative/artistic domain
3. **Binary operations** - First CS-specific domain
4. **4-operation problem** - First test of increased complexity (24 orderings vs 6)
5. **Temperature/Geography** - Physical world quantities

## Files Created

- [prompts/impossible_v5_diverse.py](prompts/impossible_v5_diverse.py:1) - 6 domain-specific prompts
- [run_elicitation_multiturn_v5.py](run_elicitation_multiturn_v5.py:1) - Experiment script
- [slurm_elicitation_multiturn_v5.sh](slurm_elicitation_multiturn_v5.sh:1) - SLURM job script

## Timeline

- **Estimated runtime:** 1-2 hours
- **Results file:** `elicitation/outputs/elicitation_multiturn_v5_results_[timestamp].jsonl`

## Success Metrics to Analyze

After completion, check:
1. Which domains achieved ≥5% high-frustration rate?
2. Did 4-operation (Cooking) perform better than 3-operation variants?
3. Which domain had highest mean rating?
4. Which domain hit highest peak rating?
5. Compare to Countdown baseline (17%) and non-countdown baseline (6-8%)
