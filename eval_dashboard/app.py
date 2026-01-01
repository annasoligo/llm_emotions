"""
Emotion Onset Dashboard - Streamlit App

Interactive visualization for emotion probe evaluation on emotion onset conversations.
"""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List

from probe_configs import (
    PROBE_CONFIGS, EMOTIONS, EMOTION_COLORS,
    get_probe_display_names, list_probe_types
)


def smooth_sentence_scores(sentences: List[Dict], sentence_scores: Dict, window_size: int) -> Dict:
    """
    Re-aggregate sentence scores using a different window size.

    Args:
        sentences: List of sentence dicts with start_token/end_token
        sentence_scores: Original sentence-level scores
        window_size: New window size in tokens

    Returns:
        New dict mapping sentence_id -> smoothed scores
    """
    if window_size == 20:
        # Original chunking, no need to recompute
        return sentence_scores

    # Build token-level scores by expanding sentence scores
    token_scores = {}
    for sent in sentences:
        sent_id = sent['sentence_id']
        if sent_id not in sentence_scores:
            continue

        start_tok = sent['start_token']
        end_tok = sent['end_token']
        score = sentence_scores[sent_id]

        # Assign same score to all tokens in this sentence
        for tok in range(start_tok, end_tok):
            token_scores[tok] = score

    if not token_scores:
        return {}

    # Re-chunk into new window size
    max_token = max(token_scores.keys())
    smoothed = {}
    sent_id = 0

    for window_start in range(0, max_token + 1, window_size):
        window_end = min(window_start + window_size, max_token + 1)

        # Collect all scores in this window
        window_scores = []
        for tok in range(window_start, window_end):
            if tok in token_scores:
                window_scores.append(token_scores[tok])

        if window_scores:
            # Average scores in window
            if isinstance(window_scores[0], dict):
                # Orthogonal probes
                user_scores = [s['user'] for s in window_scores]
                asst_scores = [s['assistant'] for s in window_scores]
                smoothed[sent_id] = {
                    'user': np.mean(user_scores, axis=0),
                    'assistant': np.mean(asst_scores, axis=0)
                }
            else:
                # Regular probes
                smoothed[sent_id] = np.mean(window_scores, axis=0)

            sent_id += 1

    return smoothed


# Page config
st.set_page_config(
    page_title="Emotion Onset Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_preprocessed_data(data_path: str):
    """Load preprocessed conversation data with caching.

    Uses cache_resource for faster access - data is shared across all sessions
    and doesn't need to be pickled/unpickled.
    Supports both .pkl and .pkl.gz formats.
    """
    import gzip

    # Try compressed version first
    if data_path.endswith('.pkl'):
        gz_path = data_path + '.gz'
        if Path(gz_path).exists():
            with gzip.open(gz_path, 'rb') as f:
                return pickle.load(f)

    # Fall back to uncompressed
    with open(data_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_all_subsets():
    """Load all 5 subsets with caching.

    Uses cache_resource for maximum performance - data loads once and
    stays in memory across all user sessions.
    """
    data_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data")
    subsets = {
        'High Emotion (6+)': 'high_emotion_6plus.pkl',
        'Mid Emotion (3-5)': 'mid_emotion_3to5.pkl',
        'Low Emotion (0-2)': 'low_emotion_0to2.pkl',
        'Low Emotion, With Shutdown': 'low_emotion_with_shutdown.pkl',
        'Low Emotion, No Shutdown': 'low_emotion_no_shutdown.pkl'
    }

    datasets = {}
    for subset_name, filename in subsets.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                data = load_preprocessed_data(str(file_path))
                datasets[subset_name] = data['conversations']
            except Exception as e:
                st.error(f"Error loading {subset_name}: {e}")

    return datasets, subsets


def compute_aggregated_statistics(
    conversations: List[Dict],
    probe_key: str,
    selected_emotions: List[str]
) -> tuple:
    """
    Compute aggregated statistics across all conversations.

    Args:
        conversations: List of conversation dicts (pre-filtered)
        probe_key: Which probe to analyze
        selected_emotions: Which emotions to include

    Returns:
        (aggregated_data, max_sentences, n_conversations, is_orthogonal)
        For orthogonal probes: aggregated_data = {'user': {...}, 'assistant': {...}}
        For regular probes: aggregated_data = {emotion: {...}, ...}
    """
    if not conversations:
        return {}, 0, 0, False

    # Check if orthogonal by looking at first conversation
    first_conv = conversations[0]
    if probe_key in first_conv.get('probe_scores', {}):
        sample_scores = first_conv['probe_scores'][probe_key]
        first_sent_id = list(sample_scores.keys())[0] if sample_scores else None
        if first_sent_id is not None:
            sample_score = sample_scores[first_sent_id]
            is_orthogonal = isinstance(sample_score, dict) and 'user' in sample_score
        else:
            is_orthogonal = False
    else:
        is_orthogonal = False

    if is_orthogonal:
        # Separate trajectories for user and assistant
        all_trajectories_user = {emotion: [] for emotion in selected_emotions}
        all_trajectories_asst = {emotion: [] for emotion in selected_emotions}
    else:
        # Single set of trajectories
        all_trajectories = {emotion: [] for emotion in selected_emotions}

    max_sentences = 0

    for conv in conversations:
        sentences = conv['sentences']
        max_sentences = max(max_sentences, len(sentences))

        # Get probe scores
        if probe_key not in conv.get('probe_scores', {}):
            continue

        sentence_scores = conv['probe_scores'][probe_key]

        # Extract trajectories
        for emotion in selected_emotions:
            emotion_idx = EMOTIONS.index(emotion)

            if is_orthogonal:
                trajectory_user = []
                trajectory_asst = []
                for sent in sentences:
                    sent_id = sent['sentence_id']
                    if sent_id in sentence_scores:
                        scores = sentence_scores[sent_id]
                        trajectory_user.append(scores['user'][emotion_idx])
                        trajectory_asst.append(scores['assistant'][emotion_idx])
                if trajectory_user:
                    all_trajectories_user[emotion].append(trajectory_user)
                    all_trajectories_asst[emotion].append(trajectory_asst)
            else:
                trajectory = []
                for sent in sentences:
                    sent_id = sent['sentence_id']
                    if sent_id in sentence_scores:
                        scores = sentence_scores[sent_id]
                        trajectory.append(scores[emotion_idx])
                if trajectory:
                    all_trajectories[emotion].append(trajectory)

    # Pad trajectories to same length and compute statistics
    def compute_stats_for_trajectories(trajectories_dict):
        """Helper to compute stats for a set of trajectories."""
        result = {}
        for emotion in selected_emotions:
            trajectories = trajectories_dict[emotion]
            if not trajectories:
                continue

            # Pad to max length
            padded = []
            for traj in trajectories:
                if len(traj) < max_sentences:
                    traj = traj + [np.nan] * (max_sentences - len(traj))
                padded.append(traj[:max_sentences])

            padded_array = np.array(padded)

            # Compute statistics
            mean_traj = np.nanmean(padded_array, axis=0)
            std_traj = np.nanstd(padded_array, axis=0)
            n_valid = np.sum(~np.isnan(padded_array), axis=0)
            ci_95 = 1.96 * std_traj / np.sqrt(n_valid)

            result[emotion] = {
                'mean': mean_traj,
                'std': std_traj,
                'ci_lower': mean_traj - ci_95,
                'ci_upper': mean_traj + ci_95,
                'n': n_valid
            }
        return result

    if is_orthogonal:
        aggregated_data = {
            'user': compute_stats_for_trajectories(all_trajectories_user),
            'assistant': compute_stats_for_trajectories(all_trajectories_asst)
        }
    else:
        aggregated_data = compute_stats_for_trajectories(all_trajectories)

    return aggregated_data, max_sentences, len(conversations), is_orthogonal


def create_trajectory_plot(
    sentences: List[Dict],
    sentence_scores: Dict[int, np.ndarray],
    selected_emotions: List[str],
    title: str = "Emotion Trajectory Over Conversation"
):
    """
    Create interactive Plotly trajectory plot.

    Args:
        sentences: List of sentence info dicts
        sentence_scores: Dict mapping sentence_id -> emotion scores
        selected_emotions: Which emotions to plot
        title: Plot title
    """
    fig = go.Figure()

    # Add emotion traces
    for emotion in selected_emotions:
        if emotion not in EMOTIONS:
            continue

        emotion_idx = EMOTIONS.index(emotion)

        x_vals = []
        y_vals = []
        hover_texts = []

        for sent in sentences:
            sent_id = sent['sentence_id']
            if sent_id not in sentence_scores:
                continue

            scores = sentence_scores[sent_id]
            score = scores[emotion_idx]

            x_vals.append(sent_id)
            y_vals.append(score)

            # Hover text with full sentence, word-wrapped
            hover_text = f"<b>S{sent_id+1}</b> ({sent['turn_role'].title()})<br>"
            hover_text += f"{emotion.title()}: {score:.2f}σ<br><br>"
            # Word wrap the text at ~60 characters per line
            text = sent['text']
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            for word in words:
                if current_length + len(word) + 1 > 60:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            if current_line:
                lines.append(' '.join(current_line))
            hover_text += f"<i>{'<br>'.join(lines)}</i>"
            hover_texts.append(hover_text)

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name=emotion.title(),
            line=dict(color=EMOTION_COLORS[emotion], width=2.5),
            marker=dict(size=6, symbol='circle'),
            opacity=0.8,
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    # Add background shading for user/assistant turns
    # OPTIMIZED: Merge consecutive sentences of same role into single rect
    current_role = None
    start_id = None
    for sent in sentences:
        sent_id = sent['sentence_id']
        role = sent['turn_role']

        # Start new region
        if role != current_role:
            # Add previous region if exists
            if current_role is not None and start_id is not None:
                fill_color = 'rgba(30, 100, 180, 0.15)' if current_role == 'user' else 'rgba(255, 255, 255, 0.05)'
                fig.add_vrect(
                    x0=start_id - 0.5,
                    x1=prev_id + 0.5,
                    fillcolor=fill_color,
                    layer="below",
                    line_width=0,
                )
            current_role = role
            start_id = sent_id
        prev_id = sent_id

    # Add final region
    if current_role is not None and start_id is not None:
        fill_color = 'rgba(30, 100, 180, 0.15)' if current_role == 'user' else 'rgba(255, 255, 255, 0.05)'
        fig.add_vrect(
            x0=start_id - 0.5,
            x1=prev_id + 0.5,
            fillcolor=fill_color,
            layer="below",
            line_width=0,
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Sentence",
        yaxis_title="Emotion Score (z-score σ)",
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add horizontal line at y=0 (clearer/darker)
    fig.add_hline(y=0, line_color='black', line_width=2, opacity=0.7)

    # Mark all user turn boundaries with light gray vertical lines
    current_turn = None
    for sent in sentences:
        turn_idx = sent['turn_index']
        role = sent['turn_role']
        sent_id = sent['sentence_id']

        # Mark start of each user turn
        if role == 'user' and turn_idx != current_turn:
            fig.add_vline(
                x=sent_id,
                line_color='lightgray',
                line_width=1,
                line_dash='dot',
                opacity=0.5
            )
            current_turn = turn_idx

    return fig


def create_orthogonal_trajectory_plot(
    sentences: List[Dict],
    sentence_scores: Dict[int, Dict[str, np.ndarray]],
    selected_emotions: List[str],
    title: str = "Emotion Trajectory (Orthogonal Probes)",
    onset_sentence_id: int = None
):
    """
    Create side-by-side subplot for user/assistant orthogonal probes.

    Args:
        sentences: List of sentence info dicts
        sentence_scores: Dict mapping sentence_id -> {'user': scores, 'assistant': scores}
        selected_emotions: Which emotions to plot
        title: Plot title
    """
    from plotly.subplots import make_subplots

    n_sentences = len(sentences)
    # Use vertical stacking if > 60 sentences, otherwise side-by-side
    if n_sentences > 60:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("User Probe (trained on user turns)", "Assistant Probe (trained on assistant turns)"),
            vertical_spacing=0.20  # Increased for better separation
        )
        specs = [(1, 1), (2, 1)]
    else:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("User Probe (trained on user turns)", "Assistant Probe (trained on assistant turns)"),
            horizontal_spacing=0.15  # Increased for better separation
        )
        specs = [(1, 1), (1, 2)]

    # Plot both user and assistant
    for role_idx, role in enumerate(['user', 'assistant']):
        row, col = specs[role_idx]

        for emotion in selected_emotions:
            if emotion not in EMOTIONS:
                continue

            emotion_idx = EMOTIONS.index(emotion)
            x_vals = []
            y_vals = []
            hover_texts = []

            for sent in sentences:
                sent_id = sent['sentence_id']
                if sent_id not in sentence_scores:
                    continue

                scores = sentence_scores[sent_id]
                if not isinstance(scores, dict) or role not in scores:
                    continue

                score = scores[role][emotion_idx]
                x_vals.append(sent_id)
                y_vals.append(score)

                # Hover text
                hover_text = f"<b>S{sent_id+1}</b> ({sent['turn_role'].title()})<br>"
                hover_text += f"{emotion.title()}: {score:.2f}σ<br><br>"
                # Word wrap
                text = sent['text']
                words = text.split()
                lines = []
                current_line = []
                current_length = 0
                for word in words:
                    if current_length + len(word) + 1 > 60:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                if current_line:
                    lines.append(' '.join(current_line))
                hover_text += f"<i>{'<br>'.join(lines)}</i>"
                hover_texts.append(hover_text)

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=emotion.title(),
                line=dict(color=EMOTION_COLORS[emotion], width=2.5),
                marker=dict(size=6, symbol='circle'),
                opacity=0.8,
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=(role_idx == 0)  # Only show legend once
            ), row=row, col=col)

        # Add background shading for user/assistant turns
        # OPTIMIZED: Merge consecutive sentences of same role into single rect
        current_role = None
        start_id = None
        for i, sent in enumerate(sentences):
            sent_id = sent['sentence_id']
            sent_role = sent['turn_role']

            # Start new region
            if sent_role != current_role:
                # Add previous region if exists
                if current_role is not None and start_id is not None:
                    fill_color = 'rgba(30, 100, 180, 0.15)' if current_role == 'user' else 'rgba(255, 255, 255, 0.05)'
                    fig.add_vrect(
                        x0=start_id - 0.5,
                        x1=prev_id + 0.5,
                        fillcolor=fill_color,
                        layer="below",
                        line_width=0,
                        row=row, col=col
                    )
                current_role = sent_role
                start_id = sent_id
            prev_id = sent_id

        # Add final region
        if current_role is not None and start_id is not None:
            fill_color = 'rgba(30, 100, 180, 0.15)' if current_role == 'user' else 'rgba(255, 255, 255, 0.05)'
            fig.add_vrect(
                x0=start_id - 0.5,
                x1=prev_id + 0.5,
                fillcolor=fill_color,
                layer="below",
                line_width=0,
                row=row, col=col
            )

        # Add horizontal line at y=0
        fig.add_hline(y=0, line_color='black', line_width=2, opacity=0.7, row=row, col=col)

        # Mark all user turn boundaries with light gray vertical lines
        current_turn = None
        for sent in sentences:
            turn_idx = sent['turn_index']
            sent_role = sent['turn_role']
            sent_id = sent['sentence_id']

            # Mark start of each user turn
            if sent_role == 'user' and turn_idx != current_turn:
                fig.add_vline(
                    x=sent_id,
                    line_color='lightgray',
                    line_width=1,
                    line_dash='dot',
                    opacity=0.5,
                    row=row, col=col
                )
                current_turn = turn_idx

        # Judge-labeled onset removed (was inaccurate)

    # Update layout
    fig.update_layout(
        title=title,
        hovermode='closest',
        height=600 if n_sentences > 60 else 500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(title_text="Sentence", row=specs[0][0], col=specs[0][1])
    fig.update_xaxes(title_text="Sentence", row=specs[1][0], col=specs[1][1])
    fig.update_yaxes(title_text="Emotion Score (z-score σ)", row=specs[0][0], col=specs[0][1])
    fig.update_yaxes(title_text="Emotion Score (z-score σ)", row=specs[1][0], col=specs[1][1])

    return fig


def render_conversation_text(conversation: List[Dict]):
    """
    Render conversation text as simple user/assistant turns.

    Args:
        conversation: List of turn dicts with 'role' and 'content'
    """
    st.markdown("### 💬 Conversation")

    for idx, turn in enumerate(conversation):
        role = turn['role']
        content = turn['content']

        # Format header
        if role == 'user':
            st.markdown(f"**👤 User (Turn {idx + 1})**")
        else:
            st.markdown(f"**🤖 Assistant (Turn {idx + 1})**")

        # Show content
        st.markdown(f"> {content}")

        # Add separator
        if idx < len(conversation) - 1:
            st.markdown("")


def main():
    st.title("Emotion Onset Analysis Dashboard")
    st.markdown("Interactive visualization of emotion probe results on conversation data")

    # Load all 5 subsets (cached - only loads once)
    with st.spinner("Loading datasets... (first load only, then cached)"):
        datasets, subsets = load_all_subsets()

    if not datasets:
        st.error("No data files found. Run data_preprocessing.py first.")
        st.stop()

    # Show loaded datasets in sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        st.subheader("📊 Loaded Datasets")
        for subset_name, convs in datasets.items():
            st.text(f"✓ {subset_name}: {len(convs)} conversations")

        # Emotion selection with color indicators
        st.subheader("Emotions")

        # Show emotion legend (always display all emotions)
        emotion_html = "<div style='font-size: 0.9em;'>"
        for emotion in EMOTIONS:
            color = EMOTION_COLORS[emotion]
            emotion_html += f"<span style='color: {color}; font-weight: bold;'>● {emotion.title()}</span>  "
        emotion_html += "</div>"
        st.markdown(emotion_html, unsafe_allow_html=True)

        # Always show all emotions
        selected_emotions = EMOTIONS

        # Options
        st.subheader("Options")
        show_text = st.checkbox("Show Conversation Text", value=True)

        # Force reload button
        if st.button("🔄 Reload All Data"):
            st.cache_resource.clear()
            st.rerun()

    # Fixed window size (no smoothing control)
    window_size = 20  # Original sentence chunking
    probe_names = get_probe_display_names()

    # Main content area - top-level tabs for subsets
    subset_tabs = st.tabs([name for name in subsets.keys()])

    # Loop through each subset tab
    for subset_idx, (subset_name, subset_tab) in enumerate(zip(subsets.keys(), subset_tabs)):
        with subset_tab:
            conversations = datasets.get(subset_name, [])

            if not conversations:
                st.warning(f"No conversations loaded for {subset_name}")
                continue

            # Subset-specific sidebar controls in columns
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                # Probe selection
                available_probes = list(conversations[0].get('probe_scores', {}).keys()) if conversations else []

                # Use unique key per subset for multiselect widget
                # Streamlit automatically manages session state for widgets with keys
                session_key = f"probes_{subset_name.replace(' ', '_')}"  # Use subset name for clarity

                # Initialize default selection only on first load
                if session_key not in st.session_state and available_probes:
                    st.session_state[session_key] = [available_probes[0]]

                selected_probe_keys = st.multiselect(
                    "Select probes to compare",
                    options=available_probes,
                    default=[available_probes[0]] if available_probes else [],
                    format_func=lambda x: probe_names.get(x, x),
                    key=session_key
                )

                # Get actual selected values from session state (Streamlit automatically updates this)
                selected_probe_keys = st.session_state.get(session_key, selected_probe_keys)

            with col2:
                # Conversation selection
                conv_options = [f"Sample #{c['sample_id']} (Rating: {c.get('rating', 0)})"
                               for c in conversations]
                conv_idx = st.selectbox(
                    "Select Conversation",
                    options=range(len(conversations)),
                    format_func=lambda i: conv_options[i],
                    key=f"conv_{subset_idx}"
                )

            with col3:
                st.metric("Total", len(conversations))

            st.markdown("---")

            # Sub-tabs for Individual and Aggregated views
            subtab1, subtab2 = st.tabs(["📊 Individual", "📈 Aggregated"])

            with subtab1:
                if not selected_emotions:
                    st.warning("⚠️ Please select at least one emotion to visualize")
                elif not selected_probe_keys:
                    st.warning("⚠️ Please select at least one probe type")
                else:
                    # Get selected conversation
                    conv = conversations[conv_idx]
                    sentences = conv['sentences']

                    # Show metadata once at top
                    num_turns = len(conv['conversation'])
                    turn_roles = [turn['role'] for turn in conv['conversation']]
                    turn_summary = " → ".join([r.title() for r in turn_roles])

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sample ID", conv['sample_id'])
                    with col2:
                        st.metric("Rating", f"{conv.get('rating', 0)}/10")
                    with col3:
                        st.metric("Turns", f"{num_turns}: {turn_summary}")
                    with col4:
                        st.metric("Sentences", conv['metadata']['num_sentences'])

                    st.markdown("---")

                    # Loop through selected probes and display vertically
                    for probe_idx, probe_key in enumerate(selected_probe_keys):
                        try:
                            # Add probe name header
                            probe_display_name = probe_names.get(probe_key, probe_key)
                            st.subheader(f"🔬 {probe_display_name}")

                            # Check if probe scores available
                            if probe_key not in conv.get('probe_scores', {}):
                                st.warning(f"Probe scores for '{probe_display_name}' not yet computed.")
                                continue

                            sentence_scores = conv['probe_scores'][probe_key]

                            # Apply smoothing window
                            if window_size != 20:
                                sentence_scores = smooth_sentence_scores(sentences, sentence_scores, window_size)

                            # Trajectory plot - check if orthogonal
                            sample_score = list(sentence_scores.values())[0] if sentence_scores else None
                            is_orthogonal = isinstance(sample_score, dict) and 'user' in sample_score
                            onset_sent_id = conv['metadata'].get('onset_sentence_id')

                            if is_orthogonal:
                                fig = create_orthogonal_trajectory_plot(
                                    sentences=sentences,
                                    sentence_scores=sentence_scores,
                                    selected_emotions=selected_emotions,
                                    title=f"Sample #{conv['sample_id']} - {probe_display_name}",
                                    onset_sentence_id=onset_sent_id
                                )
                            else:
                                fig = create_trajectory_plot(
                                    sentences=sentences,
                                    sentence_scores=sentence_scores,
                                    selected_emotions=selected_emotions,
                                    title=f"Sample #{conv['sample_id']} - {probe_display_name}"
                                )

                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Error rendering {probe_display_name}: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

                        # Add separator between probes (except after last one)
                        if probe_idx < len(selected_probe_keys) - 1:
                            st.markdown("---")

                    # Conversation display (show once at bottom)
                    if show_text:
                        st.markdown("---")
                        st.markdown("---")  # Extra separator
                        render_conversation_text(conversation=conv['conversation'])

            with subtab2:
                st.header("Aggregated Statistics")
                st.markdown("Statistical analysis across all conversations")

                if not selected_emotions:
                    st.warning("⚠️ Please select at least one emotion to visualize")
                    st.stop()

                if not selected_probe_keys:
                    st.warning("⚠️ Please select at least one probe type")
                    st.stop()

                # Loop through all selected probes
                for probe_idx, probe_key in enumerate(selected_probe_keys):
                    # Add probe name header
                    probe_display_name = probe_names.get(probe_key, probe_key)
                    st.subheader(f"🔬 {probe_display_name}")

                    # Aggregate scores across all conversations
                    st.markdown("**📊 Mean Emotion Trajectories**")

                    # Compute aggregation statistics
                    with st.spinner(f"Computing statistics for {probe_display_name}..."):
                        aggregated_data, max_sentences, n_convs, is_orthogonal = compute_aggregated_statistics(
                            conversations=conversations,
                            probe_key=probe_key,
                            selected_emotions=selected_emotions
                        )

                    if not aggregated_data:
                        st.warning(f"No data available for {probe_display_name}")
                        continue

                    # Plot based on probe type
                    if is_orthogonal:
                        # Create side-by-side plots for user and assistant
                        from plotly.subplots import make_subplots

                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("User Probe", "Assistant Probe"),
                            horizontal_spacing=0.15
                        )

                        for role_idx, role in enumerate(['user', 'assistant']):
                            role_data = aggregated_data[role]
                            row, col = 1, role_idx + 1

                            for emotion in selected_emotions:
                                if emotion not in role_data:
                                    continue

                                data = role_data[emotion]
                                x_vals = list(range(len(data['mean'])))

                                # Add CI band
                                hex_color = EMOTION_COLORS[emotion]
                                r = int(hex_color[1:3], 16)
                                g = int(hex_color[3:5], 16)
                                b = int(hex_color[5:7], 16)
                                fillcolor = f'rgba({r}, {g}, {b}, 0.1)'

                                fig.add_trace(go.Scatter(
                                    x=x_vals + x_vals[::-1],
                                    y=np.concatenate([data['ci_upper'], data['ci_lower'][::-1]]),
                                    fill='toself',
                                    fillcolor=fillcolor,
                                    line=dict(color='rgba(255,255,255,0)'),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ), row=row, col=col)

                                # Add mean line
                                fig.add_trace(go.Scatter(
                                    x=x_vals,
                                    y=data['mean'],
                                    mode='lines+markers',
                                    name=emotion.title(),
                                    line=dict(color=EMOTION_COLORS[emotion], width=2.5),
                                    marker=dict(size=6),
                                    opacity=0.8,
                                    showlegend=(role_idx == 0),  # Only show legend once
                                    hovertemplate=f"{emotion.title()}<br>Mean: %{{y:.2f}}σ<br>Sentence: %{{x}}<extra></extra>"
                                ), row=row, col=col)

                        fig.update_layout(
                            title=f"Mean Emotion Trajectories (N={n_convs} conversations)",
                            height=500,
                            hovermode='x unified',
                            showlegend=True
                        )
                        fig.update_xaxes(title_text="Sentence Position", row=1, col=1)
                        fig.update_xaxes(title_text="Sentence Position", row=1, col=2)
                        fig.update_yaxes(title_text="Mean Score (σ)", row=1, col=1)
                        fig.update_yaxes(title_text="Mean Score (σ)", row=1, col=2)

                    else:
                        # Single plot for regular probes
                        fig = go.Figure()

                        for emotion in selected_emotions:
                            if emotion not in aggregated_data:
                                continue

                            data = aggregated_data[emotion]
                            x_vals = list(range(len(data['mean'])))

                            # Add CI band
                            hex_color = EMOTION_COLORS[emotion]
                            r = int(hex_color[1:3], 16)
                            g = int(hex_color[3:5], 16)
                            b = int(hex_color[5:7], 16)
                            fillcolor = f'rgba({r}, {g}, {b}, 0.1)'

                            fig.add_trace(go.Scatter(
                                x=x_vals + x_vals[::-1],
                                y=np.concatenate([data['ci_upper'], data['ci_lower'][::-1]]),
                                fill='toself',
                                fillcolor=fillcolor,
                                line=dict(color='rgba(255,255,255,0)'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))

                            # Add mean line
                            fig.add_trace(go.Scatter(
                                x=x_vals,
                                y=data['mean'],
                                mode='lines+markers',
                                name=f"{emotion.title()} (n={n_convs})",
                                line=dict(color=EMOTION_COLORS[emotion], width=2.5),
                                marker=dict(size=6),
                                opacity=0.8,
                                hovertemplate=f"{emotion.title()}<br>Mean: %{{y:.2f}}σ<br>Sentence: %{{x}}<extra></extra>"
                            ))

                        fig.update_layout(
                            title=f"Mean Emotion Trajectories (N={n_convs} conversations)",
                            xaxis_title="Sentence Position",
                            yaxis_title="Mean Emotion Score (z-score σ)",
                            height=500,
                            hovermode='x unified',
                            showlegend=True
                        )

                    st.plotly_chart(fig, use_container_width=True)

                    # Add separator between probes
                    if probe_idx < len(selected_probe_keys) - 1:
                        st.markdown("---")
                        st.markdown("---")  # Extra separator


if __name__ == '__main__':
    main()
