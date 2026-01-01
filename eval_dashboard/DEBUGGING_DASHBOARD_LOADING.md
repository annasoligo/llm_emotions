# Dashboard Loading Issue - Debugging Summary

## Problem

User reports: "I can see the titles on the first tab but no plots or conversations and I see nothing on the other tabs"

## Root Cause Analysis

### Issue 1: Streamlit Multiselect Session State Bug

**Location:** `app.py` lines 638-657

**Problem:**
```python
# OLD CODE (BUGGY):
selected_probe_keys = st.multiselect(
    "Select probes to compare",
    options=available_probes,
    default=default_probes,  # This gets ignored after first user interaction!
    key=f"probes_{subset_idx}"
)

if not selected_probe_keys and available_probes:
    selected_probe_keys = [available_probes[0]]  # This doesn't persist!
```

**Why it fails:**
1. Once a user interacts with a multiselect and deselects all options, Streamlit stores that empty state
2. The `default` parameter is ignored in favor of the stored session state
3. The fallback logic (lines 651-652) sets a Python variable but doesn't update the widget state
4. On next render, the multiselect returns empty list from session state

**Result:** `selected_probe_keys` is empty → warnings show "Please select at least one probe type" → no plots render

**Fix Applied:**
```python
# NEW CODE (FIXED):
# Initialize session state for probe selection if not exists
session_key = f"probes_{subset_idx}"
if session_key not in st.session_state and available_probes:
    st.session_state[session_key] = [available_probes[0]]

# Ensure at least one probe is selected
if session_key in st.session_state and not st.session_state[session_key] and available_probes:
    st.session_state[session_key] = [available_probes[0]]

selected_probe_keys = st.multiselect(
    "Select probes to compare",
    options=available_probes,
    default=st.session_state.get(session_key, [available_probes[0]] if available_probes else []),
    format_func=lambda x: probe_names.get(x, x),
    key=session_key
)
```

**Benefits:**
- Explicitly manages session state
- Guarantees at least one probe is always selected
- Prevents empty multiselect state from persisting

---

## Debugging Enhancements Added

To help identify the exact point of failure, I added comprehensive debug output throughout the rendering pipeline:

### 1. Top-Level Debug Info (Always Visible)

**Location:** Line 676
```python
st.info(f"🐛 DEBUG: Emotions={len(selected_emotions)}, Probes={len(selected_probe_keys)}, Convs={len(conversations)}")
```

Shows at a glance if key variables are empty.

### 2. Detailed Debug Expander (Expanded by Default)

**Location:** Lines 678-685
```python
with st.expander("🐛 Detailed Debug Info", expanded=True):
    st.write(f"**Selected emotions:** {selected_emotions}")
    st.write(f"**Selected probe keys:** {selected_probe_keys}")
    st.write(f"**Number of conversations:** {len(conversations)}")
    if conversations:
        st.write(f"**First conv sample_id:** {conversations[0].get('sample_id')}")
        st.write(f"**First conv probe keys:** {list(conversations[0].get('probe_scores', {}).keys())}")
        st.write(f"**Current conv_idx:** {conv_idx}")
```

Shows detailed state information.

### 3. Individual Tab Debug Trail

**Location:** Throughout Individual subtab (lines 691, 698, 718, 721, 725, 730, 738, 748, 766, 768)

Debug messages at every key step:
- ✓ "Entered Individual subtab"
- ✓ "Passed validation checks, proceeding to render..."
- ✓ "Rendered metadata for sample X"
- ✓ "About to loop through N probes"
- ✓ "Processing probe 1/N: probe_key"
- ✓ "About to check probe scores..."
- ✓ "Got X sentence scores"
- ✓ "is_orthogonal=True/False, creating plot..."
- ✓ "Plot created, rendering with plotly_chart..."
- ✓ "✓ Plot rendered successfully!"

### 4. Aggregated Tab Debug Trail

**Location:** Throughout Aggregated subtab (lines 786, 798, 802)

Debug messages:
- ✓ "Entered Aggregated subtab"
- ✓ "Passed validation, looping through N probes"
- ✓ "Processing probe 1/N: probe_key"

### 5. Enhanced Error Catching

**Location:** Lines 770-773

```python
except Exception as e:
    st.error(f"❌ Error rendering {probe_display_name}: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
```

Shows full traceback if plot rendering fails.

---

## How to Use Debug Output

When you run the dashboard now, you'll see:

1. **At the top of each subset tab:**
   - Info banner: "🐛 DEBUG: Emotions=6, Probes=1, Convs=12"
   - Expanded expander with detailed state

2. **In the Individual tab:**
   - A trail of debug messages showing exactly how far execution progresses
   - If you see "Entered Individual subtab" but nothing after → validation check failed
   - If you see "About to loop through 1 probes" but no "Processing probe..." → loop not executing
   - If you see "Plot created" but no "✓ Plot rendered successfully!" → plotly_chart call hanging

3. **In the Aggregated tab:**
   - Similar debug trail
   - If it stops after spinner, the compute_aggregated_statistics() function is failing

4. **Error messages:**
   - Any exceptions will show as red error boxes with full tracebacks

---

## Expected Outcomes

### If session state fix worked:
- You should see: "🐛 DEBUG: Emotions=6, Probes=1, Convs=12" (or higher numbers)
- Plots should render successfully

### If there's still an issue:
- Debug trail will show exactly where execution stops
- Look for the LAST debug message that appears
- Check if there are any error messages or warnings

---

## Next Steps

1. **Run the dashboard** with these debug additions
2. **Check the debug output** - what's the last debug message you see?
3. **Look for errors** - are there any red error boxes or orange warnings?
4. **Report back** what you see, and we can identify the exact failure point

The debug output is extensive and should pinpoint the issue immediately.

---

## Files Modified

- `eval_dashboard/app.py`: Fixed session state handling + added comprehensive debug output

## Commit

```
4731ce5 - Add extensive debugging output to dashboard
```
