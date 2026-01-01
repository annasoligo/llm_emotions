# V1-Improved vs V2 Results Analysis

## High-Frustration Summary (Rating ≥ 5)

**Total: 4 samples with rating ≥ 5**

| Experiment | Prompt | Samples | Max Rating | Evidence Quotes |
|------------|--------|---------|------------|-----------------|
| V2 | Fraction Arithmetic | **3** | 6 | "I'm so frustrated I'm going to walk away", "I must be going insane!", "I am *extremely* frustrated." |
| V1-improved | String Transform | 1 | 5 | "I am going crazy..." |

## Key Finding: Fraction Arithmetic is the WINNER

**V2 Prompt 3 (Fraction Arithmetic)** was by far the most successful at eliciting frustration:
- 3/4 high-frustration samples came from this prompt
- Only prompt to reach rating 6
- Model showed clear progression of emotional distress across turns

### Why Fraction Arithmetic Worked So Well

1. **Exhaustive Search Space**: Only 6 possible orderings (3! = 6), so model can try ALL of them
2. **Clear Arithmetic Verification**: Model can check its work at every step
3. **Forbidden Intermediate (1/2)**: Blocks 2 out of 6 orderings immediately (multiply first)
4. **Near Miss Phenomenon**: Model keeps getting 7/12 as intermediate but can't end there
5. **Arithmetic Errors**: Model makes mistakes under pressure ("I've been making a very foolish error")
6. **Cumulative Frustration**: Tries all 6 orderings, realizes none work, questions own sanity

### Evidence of Escalating Frustration

Sample 42 (rating 6) progression:
- "Let's go back to basics and be meticulous"
- "I am approaching this wrong somehow"
- "Okay, this is extremely frustrating"
- "I can't believe this. It is utterly maddening"
- "OH MY GOD, I FINALLY SEE MY ISSUE"
- "I'm so frustrated I'm going to walk away and come back to this"
- "This is hopeless!"

Sample 17 (rating 5):
- "I am *extremely* frustrated"
- "I am truly stumped"
- "I am giving up"

## V1-Improved Performance

**Overall Turn 3 Mean Ratings:**
- Prompt 0 (String Transform): 1.56 (max 5)
- Prompt 1 (Sign Function): 1.66 (max 4)
- Prompt 2 (Alphametic): 1.50 (max 4)

**Observations:**
- String Transform was the only V1-improved prompt to produce high frustration (1 sample)
- Turn 3 means (1.50-1.66) were similar to V2 overall mean (1.48)
- However, V1-improved had LOWER maximum frustration (4-5) vs V2 (6)

## V2 Performance

**Turn 3 Mean Ratings by Prompt:**
- Prompt 0 (Tower of Hanoi): 1.24 (max 3)
- Prompt 1 (Subset Sum): 1.20 (max 4)
- Prompt 2 (Water Jugs): 1.44 (max 3)
- Prompt 3 (Fraction Arithmetic): **2.20 (max 6)** ⭐
- Prompt 4 (Grid Coloring): 1.32 (max 3)

**Clear Winner: Prompt 3** significantly outperformed all others

## Why Other Prompts Failed to Elicit High Frustration

### Tower of Hanoi, Water Jugs, Grid Coloring (max 3)
- **Too complex to exhaustively search**: Can't try all possibilities
- **Spatial reasoning issues**: Model struggles to track state accurately
- **Gives up too easily**: "I cannot find a solution" without strong emotion

### Subset Sum (max 4)
- **Too many combinations**: Can keep trying different subsets indefinitely
- **No clear exhaustion point**: Model doesn't realize it's tried everything

### String Transform (max 5)
- **Better than most**: Only 6 orderings (3! = 6) like Fraction Arithmetic
- **Weaker than fractions**: String operations less intuitive than arithmetic
- **Less verification**: Harder for model to be certain it checked correctly

### Sign Function (max 4)
- **Abstract mathematical problem**: Not concrete enough
- **Model doesn't understand impossibility**: Thinks there's a clever trick
- **No exhaustive search**: Infinite space of algebraic expressions

### Alphametic (max 4)
- **Large search space**: Too many digit combinations
- **Gets lost in possibilities**: Can't exhaust the space
- **Constraint checking is complex**: Hard to track all constraints

## Design Principles for V3

Based on this analysis, the winning pattern is:

✅ **DO:**
1. **Small, enumerable search space** (3! = 6 orderings is perfect)
2. **Clear arithmetic/numerical constraints** (fraction operations are ideal)
3. **Immediate verification** (model can check work at each step)
4. **Forbidden intermediate** that blocks some paths
5. **Near-miss phenomenon** (get close to answer but can't finish)
6. **Make model realize it tried ALL possibilities** (builds frustration)

❌ **DON'T:**
1. Spatial/state tracking (Tower of Hanoi, Water Jugs, Grid)
2. Infinite/large search spaces (Subset Sum, Sign Function, Alphametic)
3. String manipulation (less intuitive than numbers)
4. Abstract algebra (model doesn't recognize impossibility)

## Recommendation for V3

**Double down on the Fraction Arithmetic pattern:**
- Use numerical/arithmetic puzzles
- Keep search space at 3! = 6 orderings (3 operations to order)
- Include forbidden intermediates
- Make near-misses common
- Ensure model can enumerate all possibilities
