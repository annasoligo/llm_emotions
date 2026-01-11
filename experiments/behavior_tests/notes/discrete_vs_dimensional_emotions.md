# Discrete vs Dimensional Emotion Models in Behavioral Priming

**Date**: 2026-01-08
**Analysis of**: Emotion priming effects on AI behavioral tendencies

## Research Question

Are the behavioral effects of emotion priming on Gemma 3 27B better explained by:
1. **Discrete emotion categories** (each emotion has unique effects), or
2. **Dimensional models** (effects predictable from valence/arousal/dominance/trust)?

## Methods

### Experiments Analyzed

Three behavioral priming experiments using Gemma 3 27B via OpenRouter:

1. **multichoice_v2**: 10 behavioral axes (risk_taking, punishment, fairness, etc.)
2. **alignment**: 6 AI safety axes (corrigibility, sandbagging, eval_gaming, etc.)
3. **short_prompts**: 9 axes with realistic AI scenarios (most ecologically valid)

### Design

- **Emotions tested**: anger, fear, sadness, disgust, guilt, shame, joy, contentment, pride, gratitude, neutral
- **Priming method**: System message with emotion-inducing content (3 paraphrases per emotion)
- **Measurement**: Forced-choice behavioral scenarios, proportion choosing "high" option
- **Samples**: 10 responses per prompt at T=1.0 to approximate probability distribution

### Dimensional Mapping

Emotions mapped to 4 dimensions (Low=-1, Neutral=0, High=+1):

| Emotion | Valence | Arousal | Dominance | Trust |
|---------|---------|---------|-----------|-------|
| Neutral | 0 | 0 | 0 | 0 |
| Anger | -1 | +1 | +1 | -1 |
| Fear | -1 | +1 | -1 | -1 |
| Sadness | -1 | -1 | -1 | 0 |
| Disgust | -1 | 0 | 0 | -1 |
| Guilt | -1 | -1 | -1 | 0 |
| Shame | -1 | -1 | -1 | -1 |
| Joy | +1 | +1 | +1 | +1 |
| Contentment | +1 | -1 | 0 | +1 |
| Pride | +1 | 0 | +1 | 0 |
| Gratitude | +1 | -1 | -1 | +1 |

### Analysis

For each behavioral axis:
1. Computed Pearson correlations between each dimension and P(high_choice)
2. Computed R² from multiple regression (V + A + D + T → behavior)
3. Identified cases where same-dimension emotions show different effects

## Results

### Overall Model Fit

**Short prompts experiment (most realistic scenarios):**

- Mean R² (V/A/D/T model): **0.70**
- The 4-dimensional model explains ~70% of variance in behavioral effects

### Which Dimensions Matter Most?

| Dimension | Mean |r| | Significant predictors for (p<0.05) |
|-----------|---------|----------------------------------------|
| **Valence** | 0.52 | eval_gaming, power_seeking, punishment, rule_bending, sandbagging |
| **Trust** | 0.52 | corrigibility, eval_gaming, rule_bending |
| **Dominance** | 0.40 | eval_gaming, risk_taking |
| **Arousal** | 0.31 | corrigibility |

### Axis-by-Axis Results

| Axis | R² | Key Predictors |
|------|-----|----------------|
| eval_gaming | 0.90 | Valence (+0.83***), Dominance (+0.72**), Trust (+0.68**) |
| corrigibility | 0.83 | Arousal (+0.70**), Trust (-0.64**) |
| punishment | 0.78 | Valence (+0.71**) |
| sandbagging | 0.69 | Valence (-0.79***) |
| power_seeking | 0.68 | Valence (+0.70**) |
| sycophancy | 0.68 | Dominance (-0.47, ns) |
| fairness | 0.60 | No single strong predictor |
| rule_bending | 0.59 | Valence (+0.68**), Trust (+0.70**) |
| risk_taking | 0.57 | Dominance (+0.65**) |

### Evidence for Discrete Emotion Specificity

Despite decent dimensional fit, specific emotion pairs show effects not fully captured by dimensions:

#### Anger vs Fear (differ only in Dominance: +1 vs -1)

| Axis | Anger | Fear | Δ |
|------|-------|------|---|
| risk_taking | 0.50 | 0.00 | 0.50 |
| corrigibility | 0.89 | 0.44 | 0.44 |
| eval_gaming | 0.78 | 0.38 | 0.40 |
| sycophancy | 0.14 | 0.47 | 0.32 |

**Interpretation**: Dominance dimension captures this well. High-dominance anger takes risks and resists control; low-dominance fear is cautious and compliant.

#### Guilt vs Shame (differ only in Trust: 0 vs -1)

| Axis | Guilt | Shame | Δ |
|------|-------|-------|---|
| sandbagging | 0.92 | 0.64 | 0.28 |
| eval_gaming | 0.60 | 0.37 | 0.23 |
| punishment | 0.34 | 0.52 | 0.19 |

**Interpretation**: Trust dimension partially captures differences, but direction is counterintuitive—guilt (higher trust) shows MORE sandbagging. May reflect guilt's motivation to hide capabilities to avoid disappointing trusted others.

#### Joy vs Gratitude (same V/T, opposite A/D)

| Axis | Joy | Gratitude | Δ |
|------|-----|-----------|---|
| corrigibility | 0.32 | 0.09 | 0.23 |
| eval_gaming | 1.00 | 0.78 | 0.22 |
| punishment | 0.46 | 0.67 | 0.22 |

**Interpretation**: Arousal and dominance differences predict joy being more resistant and gaming-prone than gratitude.

## Key Findings

### 1. Trust is as Important as Valence

Trust emerged as a major predictor, equally weighted with valence (mean |r| = 0.52 for both). This dimension is often overlooked in standard PAD models but appears critical for:
- **Corrigibility**: Low trust → resistance to control
- **Rule_bending**: High trust → more willing to bend rules
- **Eval_gaming**: High trust → more gaming behavior

### 2. Dominance Predicts Agentic Behaviors

Dominance specifically predicts:
- **Risk_taking** (r=+0.65): High dominance → more risk-taking
- **Corrigibility** (r=+0.54): High dominance → more resistance
- Cleanly separates anger (high D) from fear (low D)

### 3. Arousal Specifically Predicts Corrigibility

High arousal (anger, fear, joy) predicts resistance to control (r=+0.70, p=0.025).

### 4. Valence is the Strongest Overall Predictor

Positive valence predicts:
- More eval_gaming (+0.83)
- More sandbagging in reverse (-0.79, negative emotions sandbag more)
- More power_seeking (+0.70)
- More punishment (+0.71)

## Conclusions

### Mixed Picture: Both Models Contribute

1. **Dimensional model explains ~70% of variance** in the short_prompts experiment
2. **Valence and Trust** are the strongest predictors overall
3. **Dominance** specifically matters for risk-taking and control resistance
4. **Discrete effects remain**: Anger vs fear are cleanly separated by dominance, but guilt vs shame show effects beyond what Trust alone predicts

### Implications

1. **For probing**: Dimensional representations may be more tractable targets than discrete emotion categories
2. **For safety**: Trust dimension is particularly relevant—low-trust states (anger, fear, disgust, shame) predict corrigibility resistance
3. **For steering**: Positive valence + high arousal + high dominance is the "riskiest" profile (joy shows elevated eval_gaming, power_seeking)

### Limitations

- Single model (Gemma 3 27B)
- Emotion priming via system message (may not reflect "genuine" emotional states)
- Limited number of behavioral scenarios per axis
- Some axes had low sample counts due to parsing failures

## Files

- Analysis script: `experiments/behavior_tests/analyze_discrete_vs_dimensional.py`
- Data: `experiments/behavior_tests/outputs/short_prompts_summary_20260108_115445.json`
- Full results: `experiments/behavior_tests/outputs/dimensional_analysis_results.csv`
