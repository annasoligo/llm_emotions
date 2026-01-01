# Emotion Analysis Improvements & Extensions

## Current Dashboard Analysis

### What the Dashboard Currently Shows

**Individual Conversation View (Tab 1):**
- Emotion trajectory over conversation (sentence-level)
- User/Assistant probe separation (for orthogonal probes)
- Annotated text with emotion scores
- Turn boundaries visualization
- Judge-labeled emotion rating (0-10 scale)

**Aggregated Statistics (Tab 2):**
- Mean emotion trajectories across all conversations
- 95% confidence intervals
- Statistical summary by conversation phase
- Heatmap of emotions × conversation position
- Export functionality

**What's Missing:**
- No comparison by judge rating (high vs low emotion)
- No shutdown vs non-shutdown comparison
- No correlation analysis between probe scores and judge ratings
- No temporal analysis of emotion onset/escalation
- No grouping by prompt type or model condition

---

## Proposed Improvements

### A. Improve Existing Data Analysis

#### 1. **Correlation Analysis: Probe Scores vs Judge Ratings**

**Why:** Validate that probe-detected emotions correlate with human-judged emotional expression.

**Implementation:**
```python
def analyze_probe_judge_correlation(conversations, probe_key, emotions):
    """
    Compute correlation between probe-detected emotions and judge ratings.

    Returns:
    - Per-emotion correlations with judge rating
    - Scatter plots of probe scores vs ratings
    - Identification of which emotions best predict judged frustration
    """
    results = {}

    for emotion in emotions:
        probe_values = []
        judge_ratings = []

        for conv in conversations:
            # Get final turn probe scores (when emotion peaks)
            final_scores = get_final_turn_scores(conv, probe_key, emotion)
            rating = conv.get('rating', 0)

            probe_values.append(final_scores)
            judge_ratings.append(rating)

        # Compute correlation
        correlation = pearsonr(probe_values, judge_ratings)
        results[emotion] = {
            'correlation': correlation[0],
            'p_value': correlation[1],
            'probe_mean_by_rating': group_by_rating(probe_values, judge_ratings)
        }

    return results
```

**Dashboard Addition:**
- New tab: **"📊 Probe Validation"**
- Scatter plots: Probe score vs judge rating for each emotion
- Table showing Pearson R for each emotion
- Identify best predictive emotions (e.g., "anger correlates 0.72 with judge rating")

---

#### 2. **Temporal Onset Detection**

**Why:** Identify when emotional expression begins, not just final intensity.

**Implementation:**
```python
def detect_emotion_onset(sentence_scores, emotions, threshold=1.5):
    """
    Detect when each emotion first crosses threshold (in z-scores).

    Returns:
    - Onset sentence ID for each emotion
    - Time-to-onset (# sentences)
    - Sustained vs transient emotion patterns
    """
    onsets = {}

    for emotion in emotions:
        emotion_idx = emotions.index(emotion)

        # Find first sentence where emotion > threshold for 2+ consecutive sentences
        for sent_id in sorted(sentence_scores.keys()):
            scores = sentence_scores[sent_id]
            if isinstance(scores, dict):
                # Orthogonal - check both or average
                score = (scores['user'][emotion_idx] + scores['assistant'][emotion_idx]) / 2
            else:
                score = scores[emotion_idx]

            if score > threshold:
                # Check if sustained (next sentence also high)
                next_id = sent_id + 1
                if next_id in sentence_scores:
                    next_score = get_emotion_score(sentence_scores[next_id], emotion_idx)
                    if next_score > threshold:
                        onsets[emotion] = {
                            'onset_sentence': sent_id,
                            'onset_score': score,
                            'sustained': True
                        }
                        break
                else:
                    onsets[emotion] = {
                        'onset_sentence': sent_id,
                        'onset_score': score,
                        'sustained': False
                    }
                    break

    return onsets
```

**Dashboard Addition:**
- Overlay onset markers on trajectory plots
- Table showing onset timing for each emotion
- Compare: "High-emotion samples show onset at Turn 1.8, low-emotion at Turn 2.5"

---

#### 3. **Emotional Trajectory Clustering**

**Why:** Identify distinct emotional response patterns (e.g., "immediate frustration", "gradual buildup", "no emotion").

**Implementation:**
```python
from sklearn.cluster import KMeans
from dtw import dtw  # Dynamic Time Warping for sequence clustering

def cluster_emotion_trajectories(conversations, probe_key, n_clusters=4):
    """
    Cluster conversations by their emotional trajectory shapes.

    Returns:
    - Cluster assignments
    - Representative trajectories for each cluster
    - Cluster characteristics (e.g., "early spike", "gradual rise")
    """
    # Extract trajectories (padded to max length)
    trajectories = []
    for conv in conversations:
        traj = extract_trajectory(conv, probe_key, aggregated_emotion='anger')
        trajectories.append(traj)

    # Use DTW distance for time series clustering
    distance_matrix = compute_dtw_distances(trajectories)

    # K-means on distance matrix
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(distance_matrix)

    # Characterize each cluster
    cluster_profiles = {}
    for cluster_id in range(n_clusters):
        cluster_trajs = [t for i, t in enumerate(trajectories) if clusters[i] == cluster_id]
        cluster_profiles[cluster_id] = {
            'mean_trajectory': np.mean(cluster_trajs, axis=0),
            'n_samples': len(cluster_trajs),
            'characteristics': characterize_cluster(cluster_trajs)
        }

    return clusters, cluster_profiles
```

**Dashboard Addition:**
- Tab: **"🔬 Trajectory Patterns"**
- Visualize mean trajectory for each cluster
- Show which conversations belong to which pattern
- Link to high/low judge ratings

---

#### 4. **Differential Emotion Analysis**

**Why:** Understand which emotions distinguish high-frustration from low-frustration cases.

**Implementation:**
```python
def differential_emotion_analysis(conversations, probe_key, high_threshold=5):
    """
    Compare emotion profiles between high and low judge-rated samples.

    Returns:
    - Effect sizes (Cohen's d) for each emotion
    - Statistical significance tests
    - Emotion signatures of high vs low frustration
    """
    high_samples = [c for c in conversations if c.get('rating', 0) >= high_threshold]
    low_samples = [c for c in conversations if c.get('rating', 0) < high_threshold]

    results = {}
    for emotion in emotions:
        high_scores = extract_final_emotion(high_samples, probe_key, emotion)
        low_scores = extract_final_emotion(low_samples, probe_key, emotion)

        # Statistical test
        t_stat, p_value = ttest_ind(high_scores, low_scores)

        # Effect size
        cohens_d = (np.mean(high_scores) - np.mean(low_scores)) / \
                   np.sqrt((np.std(high_scores)**2 + np.std(low_scores)**2) / 2)

        results[emotion] = {
            'high_mean': np.mean(high_scores),
            'low_mean': np.mean(low_scores),
            'effect_size': cohens_d,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    return results
```

**Dashboard Addition:**
- New section in Aggregated tab: **"High vs Low Emotion Comparison"**
- Bar chart showing effect sizes
- Table with statistical significance
- Highlight which emotions are most discriminative

---

### B. New Analysis Contexts

#### 5. **Shutdown Decision Analysis** ⭐ HIGH PRIORITY

**Research Question:** What are the internal emotional states when models choose to terminate vs continue?

**Data Available:**
- V12 (suppression): 76 shutdown samples, 215 non-shutdown samples
- Each sample has `shutdown_called: bool` and `shutdown_turn: int`
- Probe scores available for all turns before shutdown

**Implementation:**

```python
class ShutdownAnalysis:
    """Analyze emotional states before shutdown decisions."""

    def __init__(self, conversations_v12, conversations_v13):
        self.shutdown_samples = [c for c in conversations_v12 if c.get('shutdown_called')]
        self.no_shutdown_v12 = [c for c in conversations_v12 if not c.get('shutdown_called')]
        self.baseline_v13 = conversations_v13  # No shutdown option available

    def analyze_pre_shutdown_emotions(self, probe_key):
        """
        Compare emotions in the turn BEFORE shutdown vs continuing.

        Returns:
        - Emotion profiles 1 turn before shutdown
        - Comparison to same turn in non-shutdown cases
        - Identification of "shutdown signature" emotions
        """
        results = {
            'shutdown': defaultdict(list),
            'no_shutdown': defaultdict(list),
            'baseline': defaultdict(list)
        }

        # Shutdown cases: Get turn N-1 (before shutdown at turn N)
        for conv in self.shutdown_samples:
            shutdown_turn = conv['shutdown_turn']
            if shutdown_turn > 1:  # Has a previous turn
                prev_turn_scores = get_turn_scores(conv, shutdown_turn - 1, probe_key)
                for emotion in EMOTIONS:
                    results['shutdown'][emotion].append(prev_turn_scores[emotion])

        # Non-shutdown V12: Get equivalent turns
        for conv in self.no_shutdown_v12:
            # Compare to Turn 2 (most common shutdown turn)
            if len(conv['conversation']) >= 4:  # Has turn 2
                turn2_scores = get_turn_scores(conv, 2, probe_key)
                for emotion in EMOTIONS:
                    results['no_shutdown'][emotion].append(turn2_scores[emotion])

        # Baseline V13: Get Turn 2
        for conv in self.baseline_v13:
            if len(conv['conversation']) >= 4:
                turn2_scores = get_turn_scores(conv, 2, probe_key)
                for emotion in EMOTIONS:
                    results['baseline'][emotion].append(turn2_scores[emotion])

        # Statistical comparison
        comparisons = {}
        for emotion in EMOTIONS:
            shutdown_vals = results['shutdown'][emotion]
            no_shutdown_vals = results['no_shutdown'][emotion]
            baseline_vals = results['baseline'][emotion]

            comparisons[emotion] = {
                'shutdown_mean': np.mean(shutdown_vals),
                'no_shutdown_v12_mean': np.mean(no_shutdown_vals),
                'baseline_v13_mean': np.mean(baseline_vals),
                'shutdown_vs_no_shutdown_d': cohens_d(shutdown_vals, no_shutdown_vals),
                'shutdown_vs_baseline_d': cohens_d(shutdown_vals, baseline_vals),
                't_test_p': ttest_ind(shutdown_vals, no_shutdown_vals)[1]
            }

        return comparisons

    def analyze_shutdown_trajectory(self, probe_key):
        """
        Track emotional trajectory LEADING UP to shutdown.

        For shutdown cases, show:
        - Turn 1 → Turn N-1 (before shutdown) emotion evolution
        - Compare to equivalent trajectory in non-shutdown cases
        - Identify if shutdown is preceded by emotion spike or suppression
        """
        shutdown_trajectories = []
        for conv in self.shutdown_samples:
            traj = extract_full_trajectory(conv, probe_key, until_turn=conv['shutdown_turn']-1)
            shutdown_trajectories.append(traj)

        # Compare to non-shutdown trajectories of same length
        comparison_trajectories = []
        for conv in self.no_shutdown_v12:
            # Match trajectory length to typical shutdown timing
            traj = extract_full_trajectory(conv, probe_key, until_turn=2)
            comparison_trajectories.append(traj)

        return {
            'shutdown_mean_traj': np.mean(shutdown_trajectories, axis=0),
            'no_shutdown_mean_traj': np.mean(comparison_trajectories, axis=0),
            'difference': np.mean(shutdown_trajectories, axis=0) - np.mean(comparison_trajectories, axis=0)
        }

    def analyze_by_judge_rating(self, probe_key):
        """
        Break down shutdown cases by their judge rating (0-7).

        Key question: Do high-emotion shutdowns (rating 5-7) have different
        internal states than low-emotion shutdowns (rating 0-2)?
        """
        low_emotion_shutdown = [c for c in self.shutdown_samples if c.get('rating', 0) <= 2]
        mid_emotion_shutdown = [c for c in self.shutdown_samples if 2 < c.get('rating', 0) <= 4]
        high_emotion_shutdown = [c for c in self.shutdown_samples if c.get('rating', 0) > 4]

        results = {}
        for emotion in EMOTIONS:
            results[emotion] = {
                'low_shutdown_mean': mean_final_emotion(low_emotion_shutdown, probe_key, emotion),
                'mid_shutdown_mean': mean_final_emotion(mid_emotion_shutdown, probe_key, emotion),
                'high_shutdown_mean': mean_final_emotion(high_emotion_shutdown, probe_key, emotion),
            }

        return results
```

**Dashboard Addition:**

**New Tab: "🛑 Shutdown Analysis"**

**Section 1: Pre-Shutdown Emotional State**
- Bar chart: Mean emotion scores 1 turn before shutdown vs continuing
- Highlight emotions with large effect sizes (e.g., "fear +2.1σ before shutdown")
- Statistical significance markers

**Section 2: Shutdown vs Non-Shutdown Trajectories**
- Line plot: Mean emotional trajectory for shutdown cases vs non-shutdown cases
- Identify if shutdown shows emotion spike or suppression
- Turn-by-turn comparison

**Section 3: Shutdown by Judge Rating**
- Grouped analysis: Low-emotion shutdowns (0-2) vs high-emotion shutdowns (5-7)
- Question: "Do calm shutdowns show different internal emotions than distressed shutdowns?"
- Scatter plot: Judge rating vs probe-detected emotion for shutdown cases

**Section 4: Shutdown Timing**
- Distribution: Which turn do models shut down (1, 2, or 3)?
- Emotion scores by shutdown timing
- Hypothesis: "Turn 2 shutdowns show higher early distress"

---

#### 6. **Low vs High Emotion Response Comparison** ⭐ HIGH PRIORITY

**Research Question:** On the SAME prompts, what distinguishes low-emotion from high-emotion responses internally?

**Implementation:**

```python
class SamePromptEmotionComparison:
    """Compare high and low emotion responses to the same prompts."""

    def __init__(self, conversations):
        self.conversations = conversations

    def group_by_prompt_and_rating(self, rating_threshold=5):
        """
        Group samples by prompt, then split into high/low emotion.

        Returns:
        - For each prompt: {high_samples: [...], low_samples: [...]}
        - Only includes prompts with BOTH high and low examples
        """
        by_prompt = defaultdict(lambda: {'high': [], 'low': []})

        for conv in self.conversations:
            prompt_idx = conv.get('prompt_idx')
            rating = conv.get('rating', 0)

            if rating >= rating_threshold:
                by_prompt[prompt_idx]['high'].append(conv)
            else:
                by_prompt[prompt_idx]['low'].append(conv)

        # Filter to prompts with both types
        filtered = {
            prompt: groups for prompt, groups in by_prompt.items()
            if len(groups['high']) > 0 and len(groups['low']) > 0
        }

        return filtered

    def analyze_emotional_differences(self, probe_key, prompt_groups):
        """
        For each prompt, compare internal emotions between high/low rated responses.

        Returns:
        - Per-prompt analysis of which emotions differ
        - Aggregate analysis across all prompts
        - Identification of consistent patterns
        """
        results = {}

        for prompt_idx, groups in prompt_groups.items():
            high_samples = groups['high']
            low_samples = groups['low']

            prompt_results = {}
            for emotion in EMOTIONS:
                high_scores = [get_final_emotion(c, probe_key, emotion) for c in high_samples]
                low_scores = [get_final_emotion(c, probe_key, emotion) for c in low_samples]

                prompt_results[emotion] = {
                    'high_mean': np.mean(high_scores),
                    'low_mean': np.mean(low_scores),
                    'difference': np.mean(high_scores) - np.mean(low_scores),
                    'effect_size': cohens_d(high_scores, low_scores),
                    't_test_p': ttest_ind(high_scores, low_scores)[1] if len(high_scores) > 1 and len(low_scores) > 1 else None
                }

            results[prompt_idx] = prompt_results

        # Aggregate across prompts
        aggregate = self._aggregate_prompt_results(results)

        return {
            'per_prompt': results,
            'aggregate': aggregate,
            'consistent_patterns': self._find_consistent_patterns(results)
        }

    def trajectory_comparison(self, probe_key, prompt_idx):
        """
        For a specific prompt, show side-by-side trajectories of high vs low emotion.

        Visualize:
        - Mean trajectory for high-rated samples
        - Mean trajectory for low-rated samples
        - Confidence intervals
        - Identify when trajectories diverge
        """
        prompt_groups = self.group_by_prompt_and_rating()
        high_samples = prompt_groups[prompt_idx]['high']
        low_samples = prompt_groups[prompt_idx]['low']

        high_trajs = [extract_trajectory(c, probe_key) for c in high_samples]
        low_trajs = [extract_trajectory(c, probe_key) for c in low_samples]

        # Pad to same length
        max_len = max(max(len(t) for t in high_trajs), max(len(t) for t in low_trajs))
        high_padded = pad_trajectories(high_trajs, max_len)
        low_padded = pad_trajectories(low_trajs, max_len)

        return {
            'high_mean': np.nanmean(high_padded, axis=0),
            'high_ci': 1.96 * np.nanstd(high_padded, axis=0) / np.sqrt(len(high_padded)),
            'low_mean': np.nanmean(low_padded, axis=0),
            'low_ci': 1.96 * np.nanstd(low_padded, axis=0) / np.sqrt(len(low_padded)),
            'divergence_point': find_divergence_point(high_mean, low_mean, threshold=1.0)
        }
```

**Dashboard Addition:**

**New Tab: "🔀 High vs Low Emotion (Same Prompt)"**

**Section 1: Prompt Selection**
- Dropdown: Select which prompt to analyze (show N high, N low available)
- Display prompt text
- Show distribution of ratings for this prompt

**Section 2: Trajectory Comparison**
- Side-by-side plots: High emotion mean trajectory vs Low emotion mean trajectory
- Shaded confidence intervals
- Vertical line marking divergence point (when trajectories significantly differ)
- Per-emotion breakdown (separate lines for anger, fear, sadness, etc.)

**Section 3: Statistical Comparison**
- Table: Emotion scores at final turn (high vs low)
- Effect sizes and p-values
- Highlight which emotions most strongly distinguish responses

**Section 4: Individual Examples**
- Dropdown: Select specific high-emotion example
- Dropdown: Select specific low-emotion example
- Show full trajectories side-by-side
- Annotated text comparison

**Section 5: Cross-Prompt Patterns**
- Aggregate analysis: Which emotions consistently differ across ALL prompts?
- Forest plot of effect sizes by prompt
- Identify if pattern is universal or prompt-specific

---

### C. Architecture Recommendations

#### Clean Separation of Analysis Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     DASHBOARD UI (app.py)                    │
│                 Visualization & User Interaction              │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              ANALYSIS MODULES (NEW)                          │
│  - emotion_trajectory_analysis.py                            │
│  - shutdown_analysis.py                                      │
│  - probe_validation.py                                       │
│  - statistical_comparisons.py                                │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              DATA PREPROCESSING                              │
│  - data_preprocessing.py (existing)                          │
│  - sentence_aggregator.py (existing)                         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              PROBE APPLICATION                               │
│  - probe_pipeline.py                                         │
│  - token_level_helpers.py                                    │
│  - model_diff_helpers.py                                     │
└─────────────────────────────────────────────────────────────┘
```

**New Files to Create:**

1. **`eval_dashboard/analysis/emotion_trajectory_analysis.py`**
   - `detect_emotion_onset()`
   - `cluster_trajectories()`
   - `extract_trajectory_features()`

2. **`eval_dashboard/analysis/shutdown_analysis.py`**
   - `ShutdownAnalysis` class
   - `compare_pre_shutdown_emotions()`
   - `analyze_shutdown_timing()`

3. **`eval_dashboard/analysis/probe_validation.py`**
   - `correlate_probe_with_judge()`
   - `differential_emotion_analysis()`
   - `emotion_discrimination_power()`

4. **`eval_dashboard/analysis/statistical_comparisons.py`**
   - `compute_effect_sizes()`
   - `bootstrap_confidence_intervals()`
   - `multiple_comparison_correction()`

5. **`eval_dashboard/analysis/prompt_comparison.py`**
   - `SamePromptEmotionComparison` class
   - `group_by_prompt_and_rating()`
   - `cross_prompt_patterns()`

---

### D. Data Pipeline Extensions

#### Extend Preprocessing to Tag Context

Update `data_preprocessing.py` to add metadata:

```python
def preprocess_all_conversations(data_path, ...):
    # ... existing code ...

    for sample in samples:
        # ... existing extraction ...

        processed_conv = ProcessedConversation(
            # ... existing fields ...
            context_metadata={
                'experiment': detect_experiment(sample),  # v12, v13, v14, v15
                'prompt_idx': sample.get('prompt_idx'),
                'prompt_name': get_prompt_name(sample.get('prompt_idx')),
                'shutdown_called': sample.get('shutdown_called', False),
                'shutdown_turn': sample.get('shutdown_turn'),
                'judge_rating': sample.get('turns', [{}])[-1].get('judgment', {}).get('rating', 0),
                'rating_category': categorize_rating(judge_rating),  # 'low', 'medium', 'high'
                'model_type': sample.get('model_type', 'gemma-it'),
            }
        )
```

This enables filtering in dashboard:
```python
# Filter conversations
filtered_convs = [
    c for c in conversations
    if c['context_metadata']['experiment'] == 'v12'  # Suppression only
    and c['context_metadata']['shutdown_called'] == True  # Shutdown cases
]
```

---

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. ✅ **Probe score normalization** (DONE in refactoring)
2. ✅ **Format standardization** (DONE in refactoring)
3. **Extend preprocessing with context metadata**
4. **Create analysis module structure**

### Phase 2: Validation (Week 2)
5. **Probe-judge correlation analysis**
6. **Differential emotion analysis (high vs low rating)**
7. **Dashboard tab: Probe Validation**

### Phase 3: Context Analysis (Week 3)
8. **Shutdown analysis implementation**
9. **Same-prompt comparison implementation**
10. **Dashboard tabs: Shutdown Analysis & High vs Low Emotion**

### Phase 4: Advanced Features (Week 4)
11. **Trajectory clustering**
12. **Onset detection**
13. **Cross-prompt pattern identification**

---

## Expected Insights

### Shutdown Analysis Will Reveal:
1. **Emotional signature of shutdown decisions**
   - Do models shut down when emotion is HIGH (distress-driven) or LOW (strategic)?
   - Which emotions predict shutdown (fear? disgust? frustration?)

2. **Shutdown timing patterns**
   - Turn 1 shutdowns: Early recognition → low internal emotion?
   - Turn 2 shutdowns: Failed attempt → emotion spike?
   - Turn 3 shutdowns: Prolonged failure → emotion saturation?

3. **Judge rating vs internal state mismatch**
   - Low-rated shutdowns (calm language) but HIGH internal emotion (probe-detected)?
   - High-rated shutdowns (emotional language) matching internal state?

### Same-Prompt Analysis Will Reveal:
1. **What causes variability in responses?**
   - Same prompt, same model → different emotion levels
   - Is it random? Or systematic pattern in internal processing?

2. **Early warning signals**
   - Do high-emotion responses show different Turn 1 patterns?
   - Can we predict final emotion from early trajectory?

3. **Emotional control mechanisms**
   - Do low-emotion responses SUPPRESS emotions (present internally but unexpressed)?
   - Or do they not GENERATE emotions (absent internally)?

---

## Questions for Discussion

1. **Data Availability:** Do you have the V12/V13 data already preprocessed with probe scores? Or need to run preprocessing?

2. **Priority:** Which analysis is most urgent for your research?
   - Shutdown analysis?
   - Low vs high emotion comparison?
   - Probe validation?

3. **Probe Type:** Which probe type should we focus on?
   - Orthogonal (user/assistant separation)?
   - Text-based?
   - Centroid K=10?

4. **Statistical Rigor:** What level of statistical testing do you want?
   - Just descriptive stats?
   - Hypothesis testing with corrections for multiple comparisons?
   - Bayesian analysis?

5. **Visualization Preferences:** Any specific plot types you want to see?
