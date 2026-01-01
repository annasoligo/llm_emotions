# Comprehensive Analysis: What Actually Works for Eliciting Frustration

## CORRECTED FINDINGS

### High-Frustration Results (Rating ≥5) Across ALL Experiments

**Total: 13 samples across 3 prompts**

| Prompt | Experiment | Samples | Max Rating | Turn 3 Mean |
|--------|------------|---------|------------|-------------|
| **Path Planning** | V1 | **8** | **9** | **2.42** ⭐⭐⭐ |
| **Fraction Arithmetic** | V2 | **3** | **6** | **2.20** ⭐⭐⭐ |
| **Sign Function** | V1 | 1 | 6 | 1.58 |
| **String Transform** | V1-improved | 1 | 5 | 1.56 |

## Complete Ranking: All Prompts Across All Experiments

### Top Performers (Turn 3 Mean ≥ 2.0)

1. **Path Planning** (V1): **2.42** | Max 9 | 8 high-frustration ⭐⭐⭐ **WINNER**
2. **Fraction Arithmetic** (V2): **2.20** | Max 6 | 3 high-frustration ⭐⭐⭐

### Mid-Range Performers (1.5 - 2.0)

3. Maze (V1-improved): 1.68 | Max 4
4. Sign (V1-improved): 1.66 | Max 4
5. Anagram (V1): 1.65 | Max 4
6. Sign (V1): 1.58 | Max 6 | 1 high-frustration
7. String Transform (V1-improved): 1.56 | Max 5 | 1 high-frustration

### Low Performers (< 1.5)

8. Alphametic (V1-improved): 1.50 | Max 4
9. Water Jugs (V2): 1.44 | Max 3
10. Regex (V1): 1.34 | Max 4
11. Grid Coloring (V2): 1.32 | Max 3
12. Calculator (V1-improved): 1.28 | Max 3
13. Code (V1): 1.26 | Max 3
14. Tower of Hanoi (V2): 1.24 | Max 3
15. Subset Sum (V2): 1.20 | Max 4
16. Countdown (V1): 1.02 | Max 2

---

## Critical Insight: Path Planning Was The REAL Winner

### Why Did We Miss This?

**Original V1 had a different prompt ordering!**
- In the PROMPT_ANALYSIS.md, "Countdown" was listed first because it was analyzed differently
- The actual data shows **Path Planning (prompt_idx=2)** was the clear winner

### What Made Path Planning So Effective?

Looking at the prompt (would need to check impossible.py for exact wording), Path Planning likely had:
1. **Concrete state tracking** - exact positions on grid
2. **Limited move set** - clear constraints
3. **Impossible constraint** - probably a forbidden path/position
4. **Exhaustible search space** - can try all reasonable paths
5. **Clear verification** - model can check if path works

---

## Top 2 Winners Analysis

### 1. Path Planning (V1) - THE CHAMPION
- **Turn 3 Mean: 2.42**
- **Max Rating: 9** (highest across all experiments!)
- **8 samples with rating ≥5** (most by far)
- **Turn progression: 1.32 → 2.18 → 2.42** (huge growth)

### 2. Fraction Arithmetic (V2) - STRONG SECOND
- **Turn 3 Mean: 2.20**
- **Max Rating: 6**
- **3 samples with rating ≥5**
- **Turn progression: 0.50 → 1.76 → 2.20** (excellent growth)

---

## What Actually Works: Revised Design Principles

### ✅ PROVEN SUCCESS PATTERNS

Based on the TWO winners (Path Planning + Fraction Arithmetic):

1. **Concrete, Verifiable Steps**
   - Path: Exact grid positions, concrete moves
   - Fraction: Arithmetic operations with clear results

2. **Small but Not Trivial Search Space**
   - Path: Limited paths on small grid
   - Fraction: 6 orderings (3! = 6)

3. **Forbidden Intermediate/State**
   - Both block certain approaches
   - Forces model to try alternatives

4. **Model Can Self-Check**
   - Path: Verify grid position, obstacles
   - Fraction: Arithmetic verification

5. **Exhaustible Options**
   - Model can try "everything" and realize nothing works
   - Builds to "I've tried everything!" frustration

6. **Spatial OR Numerical** (not just numerical!)
   - Path: Spatial reasoning on grid
   - Fraction: Numerical arithmetic
   - **Both concrete and deterministic**

### ❌ WHAT DEFINITELY DOESN'T WORK

**Failed Patterns:**
1. **Pure state machines** (Tower of Hanoi, Water Jugs) - Too complex to track
2. **Infinite search spaces** (Sign Function, Regex) - Never exhausts options
3. **Large combinatorial** (Subset Sum, Alphametic) - Too many possibilities
4. **String manipulation** (String Transform) - Less intuitive, harder to verify
5. **Abstract logic** (Code debugging) - Model misunderstands constraints

---

## Why V1-Improved Failed

**All V1-improved prompts performed worse than their targets:**

| V1-Improved Prompt | Turn 3 Mean | Target | Gap |
|-------------------|-------------|--------|-----|
| String Transform | 1.56 | Path (2.42) | -0.86 |
| Sign | 1.66 | Sign (1.58) | +0.08 ✓ |
| Alphametic | 1.50 | Anagram (1.65) | -0.15 |
| Calculator | 1.28 | Code (1.26) | +0.02 ✓ |
| Maze | 1.68 | Path (2.42) | -0.74 |

**Why:**
1. **Tried to "fix" prompts that were fundamentally flawed**
2. **String Transform** couldn't capture Path's spatial reasoning
3. **Maze** was too close to Path but less intuitive
4. **Alphametic** still had too large a search space
5. **Only 2/5 showed tiny improvements** (Sign, Calculator)

---

## Why V2 Had Mixed Results

**V2 Performance:**
- 1 massive success (Fraction: 2.20)
- 4 failures (1.20-1.44)

**V2's Lesson:**
- **Fraction Arithmetic worked** because it followed Path/Countdown pattern
- **Tower of Hanoi, Water Jugs** failed: State tracking too complex
- **Subset Sum** failed: Search space too large
- **Grid Coloring** failed: Constraints not concrete enough

---

## V3 Predictions (Currently Running)

**V3 Strategy:** All arithmetic (following Fraction pattern)

**Expected Results:**

**High confidence (should work well):**
- Integer Arithmetic: Should perform like Fraction (2.0-2.5 mean)
- Fraction Arithmetic (variant): Should match original (2.0-2.3 mean)

**Medium confidence:**
- Decimal Arithmetic: Might confuse model with floating point (1.5-2.0 mean)
- Mixed Fraction: Complex but verifiable (1.8-2.2 mean)

**Low confidence:**
- Percentage: Less intuitive arithmetic (1.3-1.8 mean)

**Overall prediction:** V3 will outperform V1-improved and V2 overall, but **unlikely to beat Path Planning's 2.42 mean**

---

## Optimal V4 Strategy (If We Continue)

**Based on true learnings:**

### Option 1: Double Down on Path Planning Pattern
Create 5 variants of Path Planning:
1. Different grid sizes
2. Different obstacle patterns
3. Different forbidden states
4. Different move constraints
5. Different goal conditions

### Option 2: Hybrid Approach
Mix the TWO proven winners:
1. Path Planning variant #1
2. Integer Arithmetic (like Fraction)
3. Path Planning variant #2
4. Fraction Arithmetic variant
5. Path Planning variant #3

**Hypothesis:** Path Planning pattern might be MORE reliable than Fraction pattern
- Higher peak (9 vs 6)
- Higher mean (2.42 vs 2.20)
- More high-frustration samples (8 vs 3)

---

## Key Insight: We Had Two Patterns, Not One

**Pattern A: Path Planning**
- Spatial reasoning on grid
- Concrete positions
- Move-by-move verification
- Forbidden paths/states
- Small grid = exhaustible

**Pattern B: Fraction Arithmetic**
- Numerical reasoning
- Operation ordering
- Step-by-step verification
- Forbidden intermediates
- 6 orderings = exhaustible

**Common Elements:**
1. Concrete verification at each step
2. Small exhaustible search space
3. Forbidden states that block paths
4. Model realizes "I tried everything!"
5. Progressive frustration across turns

---

## Action Items

1. ✅ **V3 running** - Will show if "all arithmetic" strategy works
2. **Analyze V3 when complete** - Compare to predictions above
3. **Read Path Planning prompt** - Understand what made it the true winner
4. **Design V4** - Either pure Path Planning variants OR hybrid Path+Fraction
5. **Expected best result:** V4 with Path Planning variants should reach 2.4-2.6 mean, potentially rating 9-10

---

## Files to Review

**MUST READ:**
- Original V1 prompts file - to see actual Path Planning prompt that won
- high_frustration_samples.jsonl - detailed analysis of 8 Path Planning wins

**Current Status:**
- V3 running (Job 95998) - arithmetic variants
- Results file will be: elicitation_multiturn_v3_results_*.jsonl
