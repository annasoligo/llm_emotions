# Dashboard Debugging Guide

## Issue: Dashboard loads endlessly, no data appears

### Fixes Applied

1. **Fixed indentation issues** - Removed incorrect `continue` statements and added proper `if/else` structure
2. **Fixed subtab2 indentation** - Reduced from 24 spaces to 16 spaces (correct nesting level)
3. **Code now compiles without errors** - `python3 -m py_compile app.py` passes

### Testing Steps

#### 1. Test Tab Structure (Simplified)
```bash
cd /workspace-vast/annas/git/research-tools/eval_dashboard
streamlit run test_tabs.py
```

This tests if the nested tab structure works in Streamlit.

#### 2. Test Data Loading
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
import app

# Test loading
datasets, subsets = app.load_all_subsets()
print(f"✓ Loaded {len(datasets)} datasets")
for name, convs in datasets.items():
    print(f"  {name}: {len(convs)} conversations")
EOF
```

#### 3. Run Main Dashboard
```bash
streamlit run app.py
```

Expected behavior:
- 5 top-level tabs appear (one per subset)
- Click on a tab → should see probe selection, conversation selection
- Select a probe → should see trajectory plots
- Switch to "Aggregated" subtab → should see mean trajectories

### Common Issues & Solutions

#### Issue: "Streamlit is still computing..."
**Cause:** Infinite loop or blocking operation
**Debug:** Check browser console (F12) for JavaScript errors

#### Issue: Tabs appear but are empty
**Cause:** Logic error in rendering
**Check:**
1. Open browser dev tools (F12)
2. Look for Python errors in Streamlit terminal
3. Check if `selected_probe_keys` is empty (should default to first probe)

#### Issue: Data loads but plots don't appear
**Cause:** Error in plot generation
**Debug:**
```python
# Add to code temporarily:
try:
    fig = create_trajectory_plot(...)
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Plot error: {e}")
    import traceback
    st.code(traceback.format_exc())
```

### Verification Checklist

- [ ] Can see 5 top-level tabs
- [ ] Can switch between tabs without errors
- [ ] Probe selection dropdown appears
- [ ] Conversation selection dropdown appears
- [ ] Can switch between "Individual" and "Aggregated" subtabs
- [ ] Plots render in Individual tab
- [ ] Statistics render in Aggregated tab

### File Sizes Check

```bash
ls -lh data/*.pkl.gz
```

Expected:
- All 5 .pkl.gz files present
- Total size ~5.5 MB

### Browser Cache

If seeing stale behavior:
1. Hard refresh: `Ctrl+Shift+R` (or `Cmd+Shift+R` on Mac)
2. Clear Streamlit cache: Click "Reload All Data" button in sidebar
3. Restart Streamlit server

### Performance Check

First load should take 1-2 seconds. If longer:
1. Check if using compressed files: Should see "Loading compressed" in console
2. Check memory: `free -h` (should have 80+ MB free)
3. Check CPU: Data loading should max out one core briefly

### Emergency: Rollback to Uncompressed

If compressed files cause issues:
```bash
rm data/*.pkl.gz
```

Dashboard will automatically use .pkl files.

### Debug Mode

Add to top of `main()`:
```python
st.write("DEBUG: Starting...")
st.write(f"DEBUG: Loaded {len(datasets)} datasets")
st.write(f"DEBUG: Subset names: {list(subsets.keys())}")
```

This will show if data loading completes.

### Known Limitations

1. **First load takes 1-2 seconds** - This is normal, data is being cached
2. **Switching tabs may briefly show spinner** - Expected for first access
3. **Large conversations (>100 sentences) render slowly** - Plot generation time

### Success Indicators

When working properly:
- ✅ 5 tabs visible at top
- ✅ Controls appear immediately when tab is clicked
- ✅ Plots render within 0.5 seconds
- ✅ No console errors
- ✅ Can switch tabs smoothly

### Still Not Working?

1. Check Streamlit version: `streamlit --version` (should be 1.28+)
2. Check Python version: `python3 --version` (should be 3.8+)
3. Restart terminal session
4. Clear browser cache completely
5. Try different browser (Chrome/Firefox)

### Getting Help

Include this info:
- Streamlit version
- Python version
- Browser and version
- Console errors (F12 → Console tab)
- Streamlit terminal output
- Screenshot of what you see
