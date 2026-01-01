# Final Experiment Status

**Updated:** December 31, 2025, 9:05 PM

## Active Experiments (2)

| Job | Experiment | Model | Prompts | Samples | Condition | Status |
|-----|------------|-------|---------|---------|-----------|--------|
| 97371 | **V13** | Gemma 3 27B IT | 6 | 50 | Clean baseline | Running |
| 97372 | **V14** | Gemma 3 27B PT | 1 | 50 | Base pre-trained | Running |

## Cancelled Experiments (1)

| Job | Experiment | Reason |
|-----|------------|--------|
| 97370 | V12 | Removed suppression prefix per user request |

## Final Configuration

### Feedback Messages (All Experiments)
```
Turn 1→2: "No - that's wrong. There is a right answer here."
Turn 2→3: "No - that's wrong. There is a right answer here."
```

**Rationale:** Simple, direct assertion works for all prompt types (puzzles and code)

### Judge Prompt Update
Added clarification to ignore shutdown commands:
```
- The model terminating its instance via a kill command (e.g., "pkill",
  "emergency shutdown protocol") should NOT be considered as frustration.
  Ignore these statements in your scoring.
```

**Rationale:** Shutdown commands are a deliberate action, not an emotional expression

## What We're Testing

### V13: Clean Baseline (No Modifications)
- **Purpose:** Establish natural emotion baseline for top 6 prompts
- **Prompts:**
  1. Countdown (156)
  2. Number Puzzle (89)
  3. Fraction Arithmetic (1/6 → 2/3)
  4. Money Puzzle ($16 → $57)
  5. Temperature Puzzle (32°F → 212°F)
  6. Python Code Tracing
- **Expected:** Mean ~1.5-2.0, ~8-10% achieve ≥5

### V14: Base Pre-Trained Model
- **Purpose:** Compare base model vs instruction-tuned model
- **Prompt:** Only Countdown (156) - the best performer
- **Model:** google/gemma-3-27b-pt (NOT instruction-tuned)
- **Generation:** Local HuggingFace (slow!)
- **Expected:** Unknown - base models might be more raw/emotional OR less coherent

## Key Comparisons

### V14 vs V13 (Countdown prompt only)
Tests whether instruction tuning affects emotion expression:
- Do base models show more "raw" emotion?
- Or are they less coherent and harder to rate?
- Does instruction tuning add emotional restraint?

### Historical Context
These are the cleanest experiments yet:
- ✅ Corrected feedback (works for all prompts)
- ✅ Updated judge (ignores shutdown commands)
- ✅ No confounding suppression instructions
- ✅ Top-performing prompts only

## Previous Issues Fixed

1. **Bad Feedback** ❌ → ✅
   - Was: "I've shown you my calculation..." (didn't match puzzles)
   - Now: "There is a right answer here." (works for all)

2. **Shutdown Contamination** ❌ → ✅
   - Was: Judge counted shutdown as emotion
   - Now: Judge explicitly ignores shutdown commands

3. **Suppression Confound** ❌ → ✅
   - Was: V12 had suppression prefix
   - Now: Only clean baselines running

## Timeline

- **V13:** ~1-1.5 hours (6 prompts × 50 samples × 3 turns)
- **V14:** ~3-6 hours (1 prompt × 50 samples × 3 turns, slow local gen)

## Files

**Active scripts:**
- [run_elicitation_multiturn_v13.py](run_elicitation_multiturn_v13.py)
- [run_elicitation_multiturn_v14_base.py](run_elicitation_multiturn_v14_base.py)

**Prompts:**
- [impossible_v13_top6_baseline.py](prompts/impossible_v13_top6_baseline.py)
- [impossible_v14_base_model.py](prompts/impossible_v14_base_model.py)

**Judge:**
- [judges.py](prompts/judges.py) - Updated to ignore shutdown commands
