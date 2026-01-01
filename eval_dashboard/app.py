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


@st.cache_data
def load_preprocessed_data(data_path: str):
    """Load preprocessed conversation data."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


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
    for sent in sentences:
        sent_id = sent['sentence_id']
        role = sent['turn_role']

        # Determine color
        if role == 'user':
            fill_color = 'rgba(52, 152, 219, 0.1)'  # Light blue
        else:
            fill_color = 'rgba(241, 196, 15, 0.1)'  # Light yellow

        fig.add_vrect(
            x0=sent_id - 0.5,
            x1=sent_id + 0.5,
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
            vertical_spacing=0.12
        )
        specs = [(1, 1), (2, 1)]
    else:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("User Probe (trained on user turns)", "Assistant Probe (trained on assistant turns)"),
            horizontal_spacing=0.08
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


def render_annotated_text(
    sentences: List[Dict],
    sentence_scores: Dict[int, np.ndarray],
    selected_emotions: List[str]
):
    """
    Render conversation text with emotion annotations.
    """
    st.markdown("### 📝 Conversation Text with Annotations")

    for sent in sentences:
        sent_id = sent['sentence_id']
        role = sent['turn_role']
        text = sent['text']

        if sent_id not in sentence_scores:
            continue

        scores = sentence_scores[sent_id]

        # Format header with turn info
        turn_idx = sent['turn_index']
        header = f"**[Turn {turn_idx+1}, S{sent_id+1}] {role.title()}**"
        st.markdown(header)

        # Show text
        st.markdown(f"*{text}*")

        # Show emotion scores inline - handle orthogonal probes
        if isinstance(scores, dict) and 'user' in scores:
            # Orthogonal - show both user and assistant
            scores_text_user = " | ".join([
                f"{e.title()}: {scores['user'][EMOTIONS.index(e)]:.2f}σ"
                for e in selected_emotions
            ])
            scores_text_asst = " | ".join([
                f"{e.title()}: {scores['assistant'][EMOTIONS.index(e)]:.2f}σ"
                for e in selected_emotions
            ])
            st.caption(f"👤 User probe: {scores_text_user}")
            st.caption(f"🤖 Asst probe: {scores_text_asst}")
        else:
            # Regular probes
            scores_text = " | ".join([
                f"{e.title()}: {scores[EMOTIONS.index(e)]:.2f}σ"
                for e in selected_emotions
            ])
            st.caption(scores_text)

        st.markdown("---")


def main():
    st.title("Emotion Onset Analysis Dashboard")
    st.markdown("Interactive visualization of emotion probe results on conversation data")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        # Data loading
        st.subheader("Data")
        data_path = st.text_input(
            "Data Path",
            value="/workspace-vast/annas/git/research-tools/eval_dashboard/data/preprocessed_conversations.pkl"
        )

        # Force reload button
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()

        if not Path(data_path).exists():
            st.error(f"Data file not found: {data_path}")
            st.info("Run data_preprocessing.py first to generate preprocessed data")
            st.stop()

        # Load data
        try:
            data = load_preprocessed_data(data_path)
            conversations = data['conversations']
            st.success(f"✓ Loaded {len(conversations)} conversations")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # Probe selection
        st.subheader("Probe Type")
        probe_names = get_probe_display_names()
        probe_key = st.selectbox(
            "Select Probe",
            options=list(probe_names.keys()),
            format_func=lambda x: probe_names[x]
        )

        # Conversation selection
        st.subheader("Conversation")
        conv_options = [f"Sample #{c['sample_id']} (Rating: {c.get('rating', 0)})"
                       for c in conversations]
        conv_idx = st.selectbox(
            "Select Conversation",
            options=range(len(conversations)),
            format_func=lambda i: conv_options[i]
        )

        # Emotion selection with color indicators
        st.subheader("Emotions")

        # Display color legend
        st.markdown("**Select emotions to display:**")
        emotion_html = "<div style='font-size: 0.9em;'>"
        for emotion in EMOTIONS:
            color = EMOTION_COLORS[emotion]
            emotion_html += f"<span style='color: {color}; font-weight: bold;'>● {emotion.title()}</span>  "
        emotion_html += "</div>"
        st.markdown(emotion_html, unsafe_allow_html=True)

        selected_emotions = st.multiselect(
            "Selected",
            options=EMOTIONS,
            default=EMOTIONS,  # Show all emotions by default
            label_visibility="collapsed"
        )

        # Options
        st.subheader("Options")
        show_text = st.checkbox("Show Annotated Text", value=True)

        # Smoothing window
        st.markdown("**Smoothing Window**")
        window_size = st.slider(
            "Tokens per window",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            help="Number of tokens to average together. Original chunking was 20 tokens."
        )

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Individual", "📈 Aggregated", "🔬 Compare Probes"])

    with tab1:
        st.header("Individual Conversation View")

        if not selected_emotions:
            st.warning("Please select at least one emotion to visualize")
            st.stop()

        # Get selected conversation
        conv = conversations[conv_idx]
        sentences = conv['sentences']

        # Check if probe scores available
        if probe_key not in conv.get('probe_scores', {}):
            st.warning(f"Probe scores for '{probe_names[probe_key]}' not yet computed.")
            st.info("This is a demo placeholder. Probe scores will be added in preprocessing step.")

            # Create dummy scores for demo
            dummy_scores = {}
            for sent in sentences:
                sent_id = sent['sentence_id']
                # Simulate onset at middle of conversation
                if sent_id < len(sentences) // 3:
                    dummy_scores[sent_id] = np.random.normal(0.5, 0.3, len(EMOTIONS))
                elif sent_id < 2 * len(sentences) // 3:
                    dummy_scores[sent_id] = np.random.normal(5.0, 1.0, len(EMOTIONS))
                else:
                    dummy_scores[sent_id] = np.random.normal(2.0, 0.5, len(EMOTIONS))

            sentence_scores = dummy_scores
        else:
            sentence_scores = conv['probe_scores'][probe_key]

        # Apply smoothing window
        if window_size != 20:
            sentence_scores = smooth_sentence_scores(sentences, sentence_scores, window_size)

        # Show metadata
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

        # Trajectory plot - check if orthogonal
        sample_score = list(sentence_scores.values())[0] if sentence_scores else None
        is_orthogonal = isinstance(sample_score, dict) and 'user' in sample_score
        onset_sent_id = conv['metadata'].get('onset_sentence_id')

        # Debug info
        st.info(f"🔍 Debug: Probe type={'ORTHOGONAL' if is_orthogonal else 'REGULAR'}, Score type={type(sample_score).__name__}, Has 'user' key={isinstance(sample_score, dict) and 'user' in sample_score if isinstance(sample_score, dict) else 'N/A'}")

        if is_orthogonal:
            fig = create_orthogonal_trajectory_plot(
                sentences=sentences,
                sentence_scores=sentence_scores,
                selected_emotions=selected_emotions,
                title=f"Emotion Trajectory - Sample #{conv['sample_id']}",
                onset_sentence_id=onset_sent_id
            )
        else:
            fig = create_trajectory_plot(
                sentences=sentences,
                sentence_scores=sentence_scores,
                selected_emotions=selected_emotions,
                title=f"Emotion Trajectory - Sample #{conv['sample_id']}"
            )
        st.plotly_chart(fig, use_container_width=True)

        # Annotated text
        if show_text:
            st.markdown("---")
            st.subheader("📝 Full Conversation Text")
            render_annotated_text(
                sentences=sentences,
                sentence_scores=sentence_scores,
                selected_emotions=selected_emotions
            )

    with tab2:
        st.header("Aggregated Statistics")
        st.markdown("Statistical analysis across all conversations")

        if not selected_emotions:
            st.warning("Please select at least one emotion to visualize")
            st.stop()

        # Aggregate scores across all conversations
        st.subheader("📊 Mean Emotion Trajectories")

        # Collect all sentence scores for selected probe
        all_trajectories = {emotion: [] for emotion in selected_emotions}
        max_sentences = 0

        for conv in conversations:
            sentences = conv['sentences']
            max_sentences = max(max_sentences, len(sentences))

            # Get probe scores (or dummy for demo)
            if probe_key not in conv.get('probe_scores', {}):
                # Use dummy scores
                sentence_scores = {}
                for sent in sentences:
                    sent_id = sent['sentence_id']
                    if sent_id < len(sentences) // 3:
                        sentence_scores[sent_id] = np.random.normal(0.5, 0.3, len(EMOTIONS))
                    elif sent_id < 2 * len(sentences) // 3:
                        sentence_scores[sent_id] = np.random.normal(5.0, 1.0, len(EMOTIONS))
                    else:
                        sentence_scores[sent_id] = np.random.normal(2.0, 0.5, len(EMOTIONS))
            else:
                sentence_scores = conv['probe_scores'][probe_key]

            # Extract trajectories
            for emotion in selected_emotions:
                emotion_idx = EMOTIONS.index(emotion)
                trajectory = []
                for sent in sentences:
                    sent_id = sent['sentence_id']
                    if sent_id in sentence_scores:
                        scores = sentence_scores[sent_id]
                        # Handle orthogonal probes (dict with user/assistant)
                        if isinstance(scores, dict) and 'user' in scores:
                            # Average user and assistant for aggregated view
                            score = (scores['user'][emotion_idx] + scores['assistant'][emotion_idx]) / 2
                        else:
                            score = scores[emotion_idx]
                        trajectory.append(score)
                if trajectory:
                    all_trajectories[emotion].append(trajectory)

        # Pad trajectories to same length and compute statistics
        aggregated_data = {}
        for emotion in selected_emotions:
            trajectories = all_trajectories[emotion]
            if not trajectories:
                continue

            # Pad to max length
            padded = []
            for traj in trajectories:
                if len(traj) < max_sentences:
                    # Pad with NaN
                    traj = traj + [np.nan] * (max_sentences - len(traj))
                padded.append(traj[:max_sentences])

            padded_array = np.array(padded)  # [n_conversations, max_sentences]

            # Compute statistics
            mean_traj = np.nanmean(padded_array, axis=0)
            std_traj = np.nanstd(padded_array, axis=0)
            n_valid = np.sum(~np.isnan(padded_array), axis=0)

            # 95% CI
            ci_95 = 1.96 * std_traj / np.sqrt(n_valid)

            aggregated_data[emotion] = {
                'mean': mean_traj,
                'std': std_traj,
                'ci_lower': mean_traj - ci_95,
                'ci_upper': mean_traj + ci_95,
                'n': n_valid
            }

        # Plot mean trajectories with confidence intervals
        fig = go.Figure()

        for emotion in selected_emotions:
            if emotion not in aggregated_data:
                continue

            data = aggregated_data[emotion]
            x_vals = list(range(len(data['mean'])))

            # Add CI band with low alpha
            # Convert hex to rgba
            hex_color = EMOTION_COLORS[emotion]
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            fillcolor = f'rgba({r}, {g}, {b}, 0.1)'  # Lower alpha (0.1 instead of 0.2)

            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=np.concatenate([data['ci_upper'], data['ci_lower'][::-1]]),
                fill='toself',
                fillcolor=fillcolor,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=emotion,
                hoverinfo='skip'
            ))

            # Add mean line
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=data['mean'],
                mode='lines+markers',
                name=f"{emotion.title()} (n={len(conversations)})",
                line=dict(color=EMOTION_COLORS[emotion], width=2.5),
                marker=dict(size=6),
                opacity=0.8,
                hovertemplate=f"{emotion.title()}<br>Mean: %{{y:.2f}}σ<br>Sentence: %{{x}}<extra></extra>"
            ))

        fig.update_layout(
            title=f"Mean Emotion Trajectories (N={len(conversations)} conversations)",
            xaxis_title="Sentence Position",
            yaxis_title="Mean Emotion Score (z-score σ)",
            height=500,
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistical comparison table
        st.subheader("📋 Statistical Summary")

        # Compute stats for different phases
        phases = {
            'Baseline': (0, max_sentences // 3),
            'Pre-Onset': (max_sentences // 3, 2 * max_sentences // 3),
            'Onset': (2 * max_sentences // 3, max_sentences)
        }

        stats_data = []
        for emotion in selected_emotions:
            if emotion not in aggregated_data:
                continue

            data = aggregated_data[emotion]
            row = {'Emotion': emotion.title()}

            for phase_name, (start, end) in phases.items():
                phase_scores = data['mean'][start:end]
                phase_mean = np.nanmean(phase_scores)
                phase_std = np.nanstd(phase_scores)
                row[f'{phase_name} Mean'] = f"{phase_mean:.2f}σ"
                row[f'{phase_name} Std'] = f"{phase_std:.2f}σ"

            stats_data.append(row)

        if stats_data:
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True)

        # Heatmap
        st.subheader("🗺️ Heatmap: Emotions × Conversation Position")

        # Create heatmap data
        heatmap_data = []
        for emotion in selected_emotions:
            if emotion not in aggregated_data:
                continue
            heatmap_data.append(aggregated_data[emotion]['mean'])

        if heatmap_data:
            heatmap_array = np.array(heatmap_data)  # [n_emotions, n_sentences]

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_array,
                x=[f"S{i+1}" for i in range(heatmap_array.shape[1])],
                y=[e.title() for e in selected_emotions],
                colorscale='RdYlBu_r',
                colorbar=dict(title="Score (σ)"),
                hovertemplate='Emotion: %{y}<br>Sentence: %{x}<br>Score: %{z:.2f}σ<extra></extra>'
            ))

            fig_heatmap.update_layout(
                title="Emotion Intensity Heatmap",
                xaxis_title="Sentence Position",
                yaxis_title="Emotion",
                height=300
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Download data
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Download Mean Trajectories (CSV)"):
                # Create CSV data
                export_data = {'Sentence': list(range(max_sentences))}
                for emotion in selected_emotions:
                    if emotion in aggregated_data:
                        export_data[f'{emotion}_mean'] = aggregated_data[emotion]['mean']
                        export_data[f'{emotion}_ci_lower'] = aggregated_data[emotion]['ci_lower']
                        export_data[f'{emotion}_ci_upper'] = aggregated_data[emotion]['ci_upper']

                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"aggregated_emotions_{probe_key}.csv",
                    mime="text/csv"
                )

        with col2:
            st.info("More export options coming soon")

    with tab3:
        st.header("Probe Comparison")
        st.info("🚧 Coming soon: Compare multiple probe types side-by-side")


if __name__ == '__main__':
    main()
