"""Quick test to verify tab structure works"""
import streamlit as st

st.title("Tab Structure Test")

# Create data
datasets = {
    'Dataset A': ['conv1', 'conv2'],
    'Dataset B': ['conv3', 'conv4'],
}
subsets = {'Dataset A': 'a.pkl', 'Dataset B': 'b.pkl'}

# Top-level tabs
subset_tabs = st.tabs([name for name in subsets.keys()])

# Loop through tabs
for subset_idx, (subset_name, subset_tab) in enumerate(zip(subsets.keys(), subset_tabs)):
    with subset_tab:
        st.write(f"You're viewing: {subset_name}")

        conversations = datasets.get(subset_name, [])

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            selected = st.multiselect(
                "Select items",
                options=["Option 1", "Option 2"],
                default=["Option 1"],
                key=f"select_{subset_idx}"
            )
        with col2:
            st.metric("Count", len(conversations))

        # Sub-tabs
        subtab1, subtab2 = st.tabs(["View 1", "View 2"])

        with subtab1:
            st.write("This is view 1")
            st.write(f"Selected: {selected}")
            st.write(f"Conversations: {conversations}")

        with subtab2:
            st.write("This is view 2")
            st.write(f"Aggregated data for {subset_name}")

st.success("✓ Tab structure test complete")
