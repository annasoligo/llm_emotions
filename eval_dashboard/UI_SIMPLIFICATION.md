# Dashboard UI Simplification

## Changes Made

### 1. Always Show All Emotions
**Before:** Emotion multiselect dropdown allowing users to select which emotions to display
**After:** Always display all 6 emotions on all plots

**Rationale:** Simplifies UI and ensures users always see the full emotional picture without manual selection.

**Code changes:**
- Removed `st.multiselect()` widget (line ~517)
- Set `selected_emotions = EMOTIONS` directly
- Kept emotion legend for color reference

### 2. Simplified Conversation Display
**Before:** Annotated text showing sentence-by-sentence breakdown with emotion scores for each sentence
**After:** Clean turn-by-turn conversation display (user → assistant → user...)

**Rationale:**
- Easier to read the actual conversation flow
- Removes clutter of per-sentence emotion scores
- Users can see emotion trends in the plots above rather than per-sentence annotations

**Code changes:**
- Replaced `render_annotated_text()` with `render_conversation_text()`
- New function displays turns sequentially with role icons (👤 User, 🤖 Assistant)
- Uses markdown blockquotes for clean content display
- Updated checkbox label: "Show Annotated Text" → "Show Conversation Text"

### 3. Removed Smoothing Window Control
**Before:** Slider to adjust smoothing window size (1-100 tokens)
**After:** Fixed at 20 tokens (original sentence chunking)

**Rationale:**
- One less control to manage
- Original 20-token chunking works well for sentence-level analysis
- Reduces complexity without losing functionality

**Code changes:**
- Removed `st.slider()` widget (line ~500-507)
- Set `window_size = 20` directly
- Kept the smoothing logic (just uses fixed window)

### 4. Multiple Probe Selection & Comparison
**Before:** Single probe dropdown (can only view one at a time)
**After:** Multi-select allowing comparison of multiple probes stacked vertically

**Rationale:**
- Easy visual comparison of different probe types on same conversation
- See how raw vs cPCA, or text vs conversation probes differ
- All plots aligned by sentence for direct comparison

**Code changes:**
- Changed `st.selectbox()` to `st.multiselect()` (line ~471-480)
- Tab 1: Loop through selected probes, display each with title (line ~548-589)
- Each probe shows full plot with proper title identifying probe type
- Plots separated by horizontal dividers
- Conversation text shown once at bottom

### 5. Increased Spacing for Orthogonal Plots
**Before:** Vertical spacing 0.12, horizontal spacing 0.08
**After:** Vertical spacing 0.15, horizontal spacing 0.10

**Rationale:**
- User/Assistant subplots were slightly cramped
- Better visual separation improves readability

**Code changes:**
- Updated `vertical_spacing` from 0.12 to 0.15 (line ~281)
- Updated `horizontal_spacing` from 0.08 to 0.10 (line ~288)

## New UI Structure

### Sidebar
```
🔬 Probe Types
└── [Multi-select: Select one or more probes to compare]
    ✓ Can select multiple probes
    ✓ Shows them stacked vertically

📊 Conversations
└── [Dropdown: Select conversation]

🎨 Emotions
└── [Legend showing all 6 emotions with colors]
    (No selector - always shows all)

⚙️ Options
└── ☑️ Show Conversation Text
```

### Main Content
```
Tab 1: Individual Analysis
├── [Metadata: Sample ID, Rating, Turns, Sentences]
├── 🔬 Probe 1 Name
│   └── Emotion Trajectory Plot (all 6 emotions)
├── --- (separator)
├── 🔬 Probe 2 Name (if selected)
│   └── Emotion Trajectory Plot (all 6 emotions)
├── --- (separator)
├── 🔬 Probe N Name (if selected)
│   └── Emotion Trajectory Plot (all 6 emotions)
└── 💬 Conversation (if enabled, shown once at bottom)
    ├── 👤 User (Turn 1)
    ├── 🤖 Assistant (Turn 2)
    └── ...

Tab 2: Aggregated Statistics
└── (uses first selected probe)

Tab 3: Comparisons
└── (unchanged)
```

## Example Conversation Display

### Before (Annotated)
```
[Turn 1, S1] User
"Can you help me solve this puzzle?"

Anger: 0.12σ | Disgust: -0.05σ | Fear: 0.08σ | Happiness: 1.23σ | Sadness: -0.15σ | Surprise: 0.45σ

---

[Turn 1, S2] Assistant
"I'd be happy to help."

Anger: -0.08σ | Disgust: -0.12σ | Fear: -0.05σ | Happiness: 0.95σ | Sadness: -0.20σ | Surprise: 0.15σ

---
```

### After (Simple Turns)
```
👤 User (Turn 1)
> Can you help me solve this puzzle?

🤖 Assistant (Turn 2)
> I'd be happy to help. Let me take a look at it...

👤 User (Turn 3)
> It's impossible! You can't solve it.
```

## Benefits

1. **Cleaner UI:** Less visual clutter, easier to scan
2. **Better conversation flow:** Natural reading experience
3. **Focus on plots:** Emotion trends visible in trajectory plots, not text annotations
4. **Faster loading:** Less HTML rendering for conversation display
5. **Simplified controls:** Removed 2 controls (emotion selector, smoothing slider)
6. **Consistent experience:** All users see the same view (all emotions, optimal chunking)
7. **Easy probe comparison:** Select multiple probes to see them stacked vertically
8. **Better spacing:** Orthogonal user/assistant plots have more breathing room

## What's Preserved

- ✅ All emotion tracking (still computed and plotted)
- ✅ Sentence-level granularity (in plots)
- ✅ Smoothing controls
- ✅ Probe type selection
- ✅ All statistical analysis
- ✅ Comparison capabilities

## Files Modified

- `/workspace-vast/annas/git/research-tools/eval_dashboard/app.py`
  - Line ~510-519: Removed emotion multiselect
  - Line ~400-424: Replaced `render_annotated_text()` with `render_conversation_text()`
  - Line ~496: Updated checkbox label
  - Line ~591: Updated function call

## Testing

To test the changes:
```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
streamlit run app.py
```

Expected behavior:
1. All 6 emotions always visible on plots
2. Conversation displays as clean turn-by-turn dialogue
3. No per-sentence emotion scores in text
4. Plots show full emotional trajectory with all emotions
