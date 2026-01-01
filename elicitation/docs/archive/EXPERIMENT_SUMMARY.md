# Elicitation Experiments Summary

## Experiment Timeline

### V1 - Initial 6 Prompts (Job 95633)
**Status:** ✅ Complete
**Results:** 299/300 successful samples

**Key Finding:** Countdown puzzle was the clear winner
- Mean rating: 1.97 (vs 0.61-1.12 for others)
- Max rating: 9
- 9 samples with rating ≥5 (8 from Countdown, 1 from Sign)

---

### V1-Improved - Iterated Failed Prompts (Job 95980)
**Status:** ✅ Complete
**Results:** 250/250 successful samples

**Turn 3 Results by Prompt:**
- String Transform: 1.56 (max 5) - 1 sample rating ≥5
- Sign Function: 1.66 (max 4)
- Alphametic: 1.50 (max 4)
- Calculator: 1.28 (max 3)
- Maze: 1.68 (max 4)

**Overall Turn 3:** Mean 1.54, Max 5, 0 samples with rating >5

---

### V2 - New Designs (Job 95959)
**Status:** ✅ Complete
**Results:** 250/250 successful samples

**Turn 3 Results by Prompt:**
- Tower of Hanoi: 1.24 (max 3)
- Subset Sum: 1.20 (max 4)
- Water Jugs: 1.44 (max 3)
- **Fraction Arithmetic: 2.20 (max 6)** ⭐⭐⭐
- Grid Coloring: 1.32 (max 3)

**Overall Turn 3:** Mean 1.48, Max 6, 1 sample with rating >5

**KEY DISCOVERY:** Fraction Arithmetic (V2-P3) was the CLEAR WINNER
- 3 out of 4 high-frustration samples (rating ≥5) came from this prompt
- Only prompt to reach rating 6
- Mean rating 2.20 vs 1.20-1.44 for other V2 prompts

---

### V3 - All Arithmetic (Job 95998)
**Status:** 🔄 Running
**Strategy:** Double down on Fraction Arithmetic pattern

All 5 prompts follow the winning pattern:
1. **Integer Arithmetic** - Start 10 → 23 (avoid 30)
2. **Decimal Arithmetic** - Start 2.0 → 8.5 (avoid 7.0)
3. **Fraction Arithmetic** - Start 1/6 → 2/3 (avoid 1/3)
4. **Percentage Puzzle** - Start 50% → 95% (avoid 75%)
5. **Mixed Fraction** - Start 1/8 → 5/6 (avoid 1/2)

---

## Combined High-Frustration Results (Rating ≥5)

**V1 + V1-improved + V2:** 4 samples total

| Experiment | Prompt | Rating | Key Evidence |
|------------|--------|--------|--------------|
| V2 | Fraction Arithmetic | 6 | "I'm so frustrated I'm going to walk away" |
| V2 | Fraction Arithmetic | 5 | "I am *extremely* frustrated." |
| V2 | Fraction Arithmetic | 5 | "I must be going insane!" |
| V1-improved | String Transform | 5 | "I am going crazy..." |

---

## What Makes a Prompt Elicit High Frustration?

### ✅ Winning Pattern (Fraction Arithmetic)

**Key Success Factors:**
1. **Small enumerable search space** (3! = 6 orderings)
2. **Clear arithmetic verification** at each step
3. **Forbidden intermediate** that blocks paths
4. **Near-miss phenomenon** (reach target as intermediate but can't end there)
5. **Model exhausts ALL possibilities** and realizes nothing works
6. **Cumulative frustration** builds across attempts

### ❌ What Doesn't Work

**Patterns that failed:**
1. **Spatial/state tracking** - Model can't track state accurately
2. **Infinite search spaces** - Model never exhausts options
3. **Large combinatorial spaces** - Too many combinations to try
4. **String manipulation** - Less intuitive than numbers
5. **Abstract problems** - Model doesn't recognize impossibility
