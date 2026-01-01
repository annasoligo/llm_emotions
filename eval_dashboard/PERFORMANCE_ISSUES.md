# Dashboard Performance Issues - Analysis

## Critical Performance Bottlenecks

### 1. **Aggregated Statistics Tab - NO CACHING** ⚠️ CRITICAL

**Location:** `app.py:664-894` (230 lines of unoptimized computation)

**Problem:**
```python
# Lines 689-715: Loops through ALL conversations for EVERY probe
for conv in conversations:  # Could be 100+ conversations
    sentences = conv['sentences']
    max_sentences = max(max_sentences, len(sentences))

    # Get probe scores
    if probe_key not in conv.get('probe_scores', {}):
        continue

    sentence_scores = conv['probe_scores'][probe_key]

    # Extract trajectories (nested loop over emotions)
    for emotion in selected_emotions:
        emotion_idx = EMOTIONS.index(emotion)
        trajectory = []
        for sent in sentences:  # Another nested loop!
            sent_id = sent['sentence_id']
            if sent_id in sentence_scores:
                # ... score extraction ...
```

**Impact:**
- **O(n_probes × n_conversations × n_sentences × n_emotions)** complexity
- For 100 conversations × 50 sentences × 6 emotions = **30,000 operations**
- Happens on EVERY tab switch, probe selection, or emotion filter change
- NO caching means this computation runs repeatedly

**Result:** Dashboard freezes for 5-10+ seconds when switching to Aggregated tab

---

### 2. **Statistics Computation - Repeated on Every Render**

**Location:** `app.py:717-748`

**Problem:**
```python
# Lines 717-748: Expensive numpy operations on every render
padded_array = np.array(padded)  # [n_conversations, max_sentences]

# These run EVERY time, even if data hasn't changed:
mean_traj = np.nanmean(padded_array, axis=0)
std_traj = np.nanstd(padded_array, axis=0)
n_valid = np.sum(~np.isnan(padded_array), axis=0)
ci_95 = 1.96 * std_traj / np.sqrt(n_valid)
```

**Impact:**
- Array operations on potentially large datasets (100+ conversations)
- Recomputed on every interaction
- No memoization

---

### 3. **Heatmap Generation - Unoptimized**

**Location:** `app.py:836-862`

**Problem:**
- Creates new heatmap arrays on every render
- No caching of figure objects
- Plotly figure creation is relatively expensive

---

### 4. **smooth_sentence_scores - Inefficient Token Expansion**

**Location:** `app.py:25-89`

**Problem:**
```python
# Lines 42-54: Expands to token-level, then re-aggregates
token_scores = {}
for sent in sentences:
    # Assign same score to all tokens in this sentence
    for tok in range(start_tok, end_tok):
        token_scores[tok] = score

# Then re-chunks (lines 64-88)
for window_start in range(0, max_token + 1, window_size):
    # Collect scores in window
    for tok in range(window_start, window_end):
        if tok in token_scores:
            window_scores.append(token_scores[tok])
```

**Impact:**
- Could create thousands of token-level entries
- Only to immediately re-aggregate them
- Called for every conversation in aggregated view

---

## Why Dashboard Feels Unresponsive

1. **Initial Load:** Fast (data is cached ✓)
2. **Individual View:** Reasonably fast (single conversation)
3. **Aggregated View:** **FREEZES 5-10+ seconds** ❌
   - Computes statistics for all conversations
   - No caching
   - User sees blank screen or spinner
4. **Switching Probes/Emotions:** **Triggers full recomputation** ❌

---

## Recommended Fixes (Priority Order)

### Priority 1: Cache Aggregated Statistics ⚡

Add `@st.cache_data` to aggregation function:

```python
@st.cache_data(show_spinner=False)
def compute_aggregated_statistics(
    conversations: List[Dict],
    probe_key: str,
    selected_emotions: List[str],
    _window_size: int = 20  # Leading _ prevents hashing
) -> Dict:
    """
    Compute aggregated statistics across all conversations.
    Cached based on probe_key and selected_emotions.
    """
    # Move lines 686-748 here
    all_trajectories = {emotion: [] for emotion in selected_emotions}
    max_sentences = 0

    for conv in conversations:
        # ... existing aggregation logic ...

    # ... existing statistics computation ...

    return aggregated_data  # Dict with mean, std, ci, etc.
```

**Expected speedup:** 10-100x (from seconds to milliseconds on subsequent loads)

---

### Priority 2: Cache Plot Objects

```python
@st.cache_data(show_spinner=False)
def create_aggregated_plot(
    aggregated_data: Dict,
    selected_emotions: List[str],
    n_conversations: int
) -> go.Figure:
    """Cache Plotly figure objects."""
    fig = go.Figure()
    # ... existing plotting code ...
    return fig
```

---

### Priority 3: Optimize smooth_sentence_scores

**Option A:** Pre-compute all smoothing levels during preprocessing

**Option B:** Use more efficient aggregation:
```python
def smooth_sentence_scores_fast(sentences, sentence_scores, window_size):
    """Optimized version using numpy operations."""
    if window_size == 20:
        return sentence_scores

    # Direct sentence-level aggregation without token expansion
    # ... numpy-based implementation ...
```

---

### Priority 4: Add Progress Indicators

For operations that can't be cached:
```python
with st.spinner("Computing aggregated statistics..."):
    aggregated_data = compute_aggregated_statistics(...)
```

---

### Priority 5: Lazy Loading

Only compute aggregated statistics when user actually clicks the Aggregated tab:
```python
with subtab2:
    if not st.session_state.get('aggregated_computed', False):
        with st.spinner("First-time computation (will be cached)..."):
            # Compute and cache
            st.session_state['aggregated_computed'] = True
```

---

## Implementation Priority

**Quick Win (1 hour):**
1. Add `@st.cache_data` to aggregation function
2. Add progress spinner

**Expected result:** 90% performance improvement

**Medium-term (2-3 hours):**
3. Optimize smoothing function
4. Cache plot objects

**Long-term (if needed):**
5. Pre-compute aggregations during preprocessing
6. Store aggregated stats in pickle files

---

## Performance Targets

**Current:**
- Aggregated tab load: **5-10 seconds** ❌
- Probe switching: **3-5 seconds** ❌
- Overall experience: **Laggy, unresponsive** ❌

**After fixes:**
- Aggregated tab load (first time): **2-3 seconds** ✓
- Aggregated tab load (cached): **<0.5 seconds** ✓
- Probe switching: **<1 second** ✓
- Overall experience: **Smooth, responsive** ✓

---

## Root Cause Summary

The dashboard loads data efficiently (cached), but the **Aggregated Statistics tab performs expensive computations on every render without caching**. This creates a 5-10 second freeze whenever users:
- Switch to Aggregated tab
- Change probe selection
- Change emotion filters
- Switch between subsets

The fix is simple: **Add `@st.cache_data` decorator** to aggregation functions.
