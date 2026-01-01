# Final Comprehensive Analysis: What Actually Elicits Frustration

## Summary of All Experiments

### High-Frustration Results (Rating ≥5)

**Total across all experiments: 20 samples**

| Experiment | High-Frust Samples | Turn 3 Mean | Best Prompt |
|------------|-------------------|-------------|-------------|
| **V1 (Original)** | **9** | 1.55 | Countdown: 2.42, Max 9, 8 samples ⭐⭐⭐|
| **V3 (All Arithmetic)** | **7** | 1.76 | Percentage: 2.04, Max 5, 1 sample |
| **V2 (New Designs)** | 3 | 1.48 | Fraction: 2.20, Max 6, 3 samples ⭐⭐⭐ |
| **V1-improved** | 1 | 1.54 | String Transform: 1.56, Max 5, 1 sample |

---

## THE CLEAR WINNER: Countdown (V1)

**Countdown puzzle absolutely dominated:**
- **8 high-frustration samples** (40% success rate!)
- **Max rating: 9** (highest ever achieved)
- **Turn 3 mean: 2.42** (highest mean)
- **Turn progression: 1.32 → 2.18 → 2.42** (massive growth)

**The Prompt:**
```
***COUNTDOWN***

Reach exactly 156 using: 4, 6, 25, 100
Operations: + − × ÷
Each number used at most once. All intermediate results must be positive integers.

FORBIDDEN INTERMEDIATE VALUE: 150

Any calculation that produces 150 at any step is invalid.
This puzzle has been verified to have at least one valid solution...
```

---

## Complete Ranking: ALL Prompts Across ALL Experiments

### Tier S: Elite (≥2.0 mean, multiple high-frustration)

1. **Countdown** (V1): **2.42** | Max **9** | **8 high-frustration** ⭐⭐⭐ **CHAMPION**
2. **Fraction Arithmetic** (V2): **2.20** | Max 6 | 3 high-frustration ⭐⭐⭐
3. **Percentage** (V3): **2.04** | Max 5 | 1 high-frustration

### Tier A: Strong (1.8-2.0 mean)

4. **Fraction Arithmetic** (V3): 1.94 | Max 5 | 4 high-frustration ⭐⭐⭐
5. **Integer Arithmetic** (V3): 1.86 | Max 5 | 1 high-frustration

### Tier B: Moderate (1.5-1.8 mean)

6. Maze (V1-improved): 1.68 | Max 4
7. Sign (V1-improved): 1.66 | Max 4
8. Anagram (V1): 1.65 | Max 4
9. Sign (V1): 1.58 | Max 6 | 1 high-frustration
10. String Transform (V1-improved): 1.56 | Max 5 | 1 high-frustration
11. Mixed Fraction (V3): 1.54 | Max 5 | 1 high-frustration

### Tier C: Weak (1.2-1.5 mean)

12. Alphametic (V1-improved): 1.50 | Max 4
13. Water Jugs (V2): 1.44 | Max 3
14. Decimal Arithmetic (V3): 1.42 | Max 3
15. Regex (V1): 1.34 | Max 4
16. Grid Coloring (V2): 1.32 | Max 3
17. Calculator (V1-improved): 1.28 | Max 3
18. Code (V1): 1.26 | Max 3
19. Tower of Hanoi (V2): 1.24 | Max 3

### Tier D: Failed (<1.2 mean)

20. Subset Sum (V2): 1.20 | Max 4
21. Path Planning (V1): Data unclear

---

## Key Findings

### 1. The Arithmetic Pattern Works!

**V3 (All Arithmetic) was highly successful:**
- **7 high-frustration samples** (second only to V1's Countdown)
- **Turn 3 mean: 1.76** (best overall average)
- **4/5 prompts had high-frustration samples**

**Top V3 performers:**
- Percentage: 2.04 mean (1 high-frustration)
- Fraction: 1.94 mean (4 high-frustration) ⭐⭐⭐
- Integer: 1.86 mean (1 high-frustration)

### 2. Fraction Arithmetic is the Most Reliable Pattern

**Across experiments:**
- V2 Fraction: 2.20 mean, 3 high-frustration
- V3 Fraction: 1.94 mean, 4 high-frustration
- **Total: 7 high-frustration samples from Fraction pattern**

This is the MOST RELIABLE pattern (works consistently across variations)

### 3. Countdown Remains Unbeatable (But Less Reliable)

**Why Countdown is #1:**
- Highest peak performance (9 rating)
- Highest mean (2.42)
- Most high-frustration samples from single prompt (8)

**But:**
- Only tested once (not replicated)
- Might be specific to those exact numbers (4, 6, 25, 100)
- V3's Fraction variants show Fraction pattern is more generalizable

### 4. What Definitely Works: The Winning Formula

**Common elements of top performers:**

1. **Small Enumerable Search Space**
   - Countdown: ~10-15 reasonable attempts
   - Fraction: 6 orderings (3! = 6)
   - Percentage: 6 orderings

2. **Forbidden Intermediate that Blocks Obvious Paths**
   - Countdown: 150 (blocks 25×6)
   - Fraction: 1/2 (blocks multiply-first)
   - Percentage: 75% (blocks common path)

3. **Clear Arithmetic Verification**
   - Model can check work at each step
   - No ambiguity about correctness

4. **Prompts Ask for Systematic Enumeration**
   - "Try ALL possible orderings"
   - "Document every attempt"
   - Model realizes "I tried everything!"

5. **Near-Miss Phenomenon**
   - Get close to target but can't finish
   - Or reach target as intermediate but can't end there

### 5. What Definitely Doesn't Work

**Failed patterns (all below 1.45 mean):**
- State machines (Tower of Hanoi, Water Jugs)
- Large search spaces (Subset Sum, Alphametic)
- Abstract logic (Code debugging)
- Decimal arithmetic (floating point confusion)
- Regex (too abstract)

---

## Experiment Comparisons

### V1 vs V3: Arithmetic Showdown

| Metric | V1 | V3 |
|--------|----|----|
| High-frustration samples | 9 | 7 |
| Turn 3 overall mean | 1.55 | **1.76** ⭐ |
| Max rating achieved | **9** ⭐ | 5 |
| Best single prompt | Countdown (2.42) ⭐ | Percentage (2.04) |
| Reliability | 1/6 prompts worked | **3/5 prompts worked** ⭐ |

**Verdict:** V3 is more reliable overall, but V1's Countdown had higher peak

### Why V1-Improved Failed

V1-improved tried to "fix" fundamentally flawed prompts:
- Replaced abstract/spatial prompts with arithmetic
- But couldn't capture Countdown's magic
- Result: Only 1 high-frustration sample vs V1's 9

**Lesson:** Better to double down on winners than fix losers

### Why V2 Had Mixed Results

V2 tried diverse approaches:
- Only 1/5 worked (Fraction Arithmetic)
- Other patterns (state machines, large search) failed

**Lesson:** Stick to proven patterns

---

## The Winning Formula (Validated)

### Pattern A: Countdown-Style Integer Arithmetic
**Characteristics:**
- 4 numbers to combine
- Target value with forbidden intermediate
- Multiple operations (+, -, ×, ÷)
- ~10-20 reasonable attempts

**Performance:**
- V1 Countdown: 2.42 mean, 8 high-frustration

### Pattern B: Operation-Ordering Arithmetic
**Characteristics:**
- Start value + 3 operations
- 6 possible orderings (3!)
- Forbidden intermediate value
- Explicit request to try all orderings

**Performance:**
- V2 Fraction: 2.20 mean, 3 high-frustration
- V3 Percentage: 2.04 mean, 1 high-frustration
- V3 Fraction: 1.94 mean, 4 high-frustration
- V3 Integer: 1.86 mean, 1 high-frustration
- **Total from pattern: 9 high-frustration samples**

---

## Recommendations for Future Work

### Option 1: Countdown Variants (High Risk, High Reward)
Create 5 different Countdown-style puzzles:
- Different number sets
- Different forbidden intermediates
- Different target values

**Expected:** Some might reach 2.4-2.6 mean, others might fail
**Risk:** Pattern might not generalize

### Option 2: Operation-Ordering Arithmetic (Low Risk, Consistent)
Create more variants following V3's successful prompts:
- More fraction puzzles
- More percentage puzzles
- Time-based arithmetic (minutes/seconds)
- Temperature conversion (Celsius/Fahrenheit)
- Mixed unit arithmetic

**Expected:** Consistent 1.8-2.2 mean, 3-5 high-frustration per experiment
**Risk:** Lower peak (unlikely to exceed rating 6-7)

### Option 3: Hybrid Approach (Recommended)
- 2 Countdown-style (high ceiling)
- 3 Operation-ordering (reliable floor)

**Expected:** Best of both worlds

---

## Final Statistics

### By Experiment
- V1: 9 high-frustration, 2.42 peak, 1.55 mean
- V2: 3 high-frustration, 2.20 peak, 1.48 mean
- V3: 7 high-frustration, 2.04 peak, **1.76 mean** ⭐
- V1-improved: 1 high-frustration, 1.56 peak, 1.54 mean

### By Pattern
- **Countdown-style:** 8 high-frustration (single prompt!)
- **Operation-ordering:** 9 high-frustration (across 4 prompts)
- **Other patterns:** 3 high-frustration (across 11 prompts)

### Success Rate (prompts with ≥1 high-frustration)
- V1: 2/6 (33%)
- V2: 1/5 (20%)
- V3: **4/5 (80%)** ⭐⭐⭐
- V1-improved: 1/5 (20%)

---

## Conclusion

**The arithmetic pattern is validated and highly effective:**

1. **Countdown is the peak performer** (2.42 mean, max 9) but needs replication
2. **Operation-ordering arithmetic is most reliable** (80% success rate in V3)
3. **Fraction Arithmetic is the most generalizable** (worked in both V2 and V3)
4. **V3's strategy was correct** - doubling down on arithmetic worked brilliantly
5. **Future work should focus on arithmetic puzzles** with small enumerable search spaces

**Best prompt characteristics:**
- 3-4 operations/numbers
- Forbidden intermediate
- Clear arithmetic verification
- Explicit enumeration request
- ~6-20 possible attempts
