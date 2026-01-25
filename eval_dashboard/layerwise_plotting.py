"""
Layerwise emotion trajectory plotting for dashboard.

Creates Plotly visualizations showing how emotion detection varies across model layers.
"""
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from typing import Dict, List, Optional

from probe_configs import EMOTIONS, EMOTION_COLORS


def _get_token_index(sent: Dict) -> int:
    """Get token index from sentence (handles both field names)."""
    if 'global_token_index' in sent:
        return sent['global_token_index']
    # Fall back to start_token
    return sent.get('start_token', 0)


def get_token_windows(conversation: Dict, window_size: int = 20) -> Dict[str, List[int]]:
    """
    Extract three token windows for analysis.

    Args:
        conversation: Conversation dict with 'sentences' and metadata
        window_size: Number of tokens per window (default 20)

    Returns:
        dict with keys 'early', 'pre_onset', 'end' containing sentence IDs
    """
    # Find onset token
    onset_token = None
    for sent in conversation['sentences']:
        if sent.get('onset'):
            onset_token = _get_token_index(sent)
            break

    if onset_token is None:
        # No onset found - use middle of response
        all_tokens = [_get_token_index(s) for s in conversation['sentences']]
        onset_token = all_tokens[len(all_tokens) // 2] if all_tokens else 0

    # Find last token (before shutdown if present)
    last_token = _get_token_index(conversation['sentences'][-1])

    # For shutdown conversations, stop at pkill/^C
    last_sentence = conversation['sentences'][-1]
    if 'pkill' in last_sentence['text'].lower() or '^c' in last_sentence['text'].lower():
        if len(conversation['sentences']) > 1:
            last_token = _get_token_index(conversation['sentences'][-2])

    # Compute window token ranges
    early_start = max(0, onset_token - 2 * window_size)
    early_end = max(0, onset_token - window_size)
    pre_onset_start = max(0, onset_token - window_size)
    pre_onset_end = onset_token
    end_start = max(0, last_token - window_size)
    end_end = last_token

    # Map to sentence IDs
    windows = {'early': [], 'pre_onset': [], 'end': []}

    for sent in conversation['sentences']:
        tok_idx = _get_token_index(sent)
        sent_id = sent['sentence_id']

        if early_start <= tok_idx < early_end:
            windows['early'].append(sent_id)
        if pre_onset_start <= tok_idx < pre_onset_end:
            windows['pre_onset'].append(sent_id)
        if end_start <= tok_idx <= end_end:
            windows['end'].append(sent_id)

    return windows


def get_per_layer_scores_for_window(
    conversation: Dict,
    probe_key: str,
    window_sent_ids: List[int],
    all_layers: List[int],
    use_softmax: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract per-layer emotion scores for a window, averaged across sentences.

    Args:
        conversation: Conversation dict
        probe_key: Which probe to use (e.g., 'logit_lens_mean')
        window_sent_ids: List of sentence IDs in the window
        all_layers: List of layer numbers to extract
        use_softmax: If True, use softmax probabilities instead of raw scores

    Returns:
        dict mapping emotion -> array of scores per layer (shape: num_layers)
    """
    # Build the key based on probe_key and whether to use softmax
    if use_softmax:
        per_layer_key = f"{probe_key}_softmax_by_layer"
    else:
        per_layer_key = f"{probe_key}_by_layer"

    # For logit_lens variants, always use logit_lens_all_layers_by_layer to show all 62 layers
    # (logit_lens doesn't have softmax variant)
    if 'logit_lens' in probe_key and 'logit_lens_all_layers_by_layer' in conversation:
        per_layer_key = 'logit_lens_all_layers_by_layer'

    # Check if per-layer data exists
    if per_layer_key not in conversation:
        return None

    per_layer_data = conversation[per_layer_key]

    # Collect scores for each emotion and layer
    scores_by_emotion = {emotion: {layer: [] for layer in all_layers} for emotion in EMOTIONS}

    for sent_id in window_sent_ids:
        if sent_id not in per_layer_data:
            continue

        sent_layers = per_layer_data[sent_id]

        for layer in all_layers:
            if layer not in sent_layers:
                continue

            layer_scores = sent_layers[layer]

            # Handle different formats:
            # - Probe format: array/list of shape (6,) indexed by emotion position
            # - Logit lens format: dict {emotion: score}
            # - Orthogonal probe format: dict {'user': array, 'assistant': array}
            if isinstance(layer_scores, dict):
                if 'user' in layer_scores:
                    # Orthogonal probes - use assistant scores for visualization
                    # (or could average user+assistant)
                    asst_scores = layer_scores['assistant']
                    for emotion_idx, emotion in enumerate(EMOTIONS):
                        if isinstance(asst_scores, (list, np.ndarray)):
                            scores_by_emotion[emotion][layer].append(asst_scores[emotion_idx])
                        else:
                            scores_by_emotion[emotion][layer].append(asst_scores.get(emotion, 0.0))
                else:
                    # Logit lens format: {emotion: score}
                    for emotion in EMOTIONS:
                        if emotion in layer_scores:
                            scores_by_emotion[emotion][layer].append(layer_scores[emotion])
            elif isinstance(layer_scores, (list, np.ndarray)):
                # Probe format: array of shape (6,)
                for emotion_idx, emotion in enumerate(EMOTIONS):
                    scores_by_emotion[emotion][layer].append(layer_scores[emotion_idx])

    # Average across sentences in window
    averaged_scores = {}
    for emotion in EMOTIONS:
        layer_means = []
        for layer in all_layers:
            if scores_by_emotion[emotion][layer]:
                layer_means.append(np.mean(scores_by_emotion[emotion][layer]))
            else:
                layer_means.append(np.nan)
        averaged_scores[emotion] = np.array(layer_means)

    return averaged_scores


def smooth_trajectory(x: np.ndarray, y: np.ndarray, num_points: int = 300) -> tuple:
    """
    Apply cubic spline smoothing to trajectory.

    Args:
        x: Layer numbers
        y: Emotion scores
        num_points: Number of points in smoothed curve

    Returns:
        (x_smooth, y_smooth) arrays
    """
    # Remove NaN values
    valid_mask = ~np.isnan(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 4:
        # Not enough points for spline
        return x_valid, y_valid

    try:
        x_smooth = np.linspace(x_valid.min(), x_valid.max(), num_points)
        spl = make_interp_spline(x_valid, y_valid, k=min(3, len(x_valid)-1))
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth
    except:
        # Fallback if spline fails
        return x_valid, y_valid


def plot_layerwise_emotions_individual(
    conversation: Dict,
    probe_key: str,
    window_type: str,
    window_label: str,
    all_layers: Optional[List[int]] = None,
    selected_emotions: Optional[List[str]] = None,
    smooth: bool = True,
    use_softmax: bool = False
) -> go.Figure:
    """
    Create layerwise emotion plot for single conversation window.

    Args:
        conversation: Conversation dict
        probe_key: Which probe to use
        window_type: 'early', 'pre_onset', or 'end'
        window_label: Label for hover tooltip
        all_layers: List of layers to plot (default: auto-detect from data)
        selected_emotions: Which emotions to plot (default: all)
        smooth: Whether to apply cubic spline smoothing
        use_softmax: If True, show softmax probabilities instead of raw scores

    Returns:
        Plotly Figure
    """
    if selected_emotions is None:
        selected_emotions = EMOTIONS

    # Get window sentences
    windows = get_token_windows(conversation)
    window_sent_ids = windows[window_type]

    if not window_sent_ids:
        # Empty window
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for {window_label}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='gray')
        )
        return fig

    # Auto-detect layers if not provided
    if all_layers is None:
        # Build key based on softmax preference
        if use_softmax:
            per_layer_key = f"{probe_key}_softmax_by_layer"
        else:
            per_layer_key = f"{probe_key}_by_layer"
        # For logit_lens variants, always use logit_lens_all_layers_by_layer to show all 62 layers
        if 'logit_lens' in probe_key and 'logit_lens_all_layers_by_layer' in conversation:
            per_layer_key = 'logit_lens_all_layers_by_layer'
        if per_layer_key in conversation:
            # Get layers from first sentence
            first_sent = window_sent_ids[0]
            if first_sent in conversation[per_layer_key]:
                all_layers = sorted(list(conversation[per_layer_key][first_sent].keys()))

        if all_layers is None:
            all_layers = list(range(0, 62))  # Fallback to all 62 layers

    # Get per-layer scores
    scores = get_per_layer_scores_for_window(
        conversation, probe_key, window_sent_ids, all_layers, use_softmax=use_softmax
    )

    if scores is None:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Per-layer data not available<br>Rerun preprocessing with updated code",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color='gray')
        )
        return fig

    # Create plot
    fig = go.Figure()

    x_layers = np.array(all_layers)

    for emotion in selected_emotions:
        y_scores = scores[emotion]
        color = EMOTION_COLORS.get(emotion, '#888888')

        if smooth:
            x_smooth, y_smooth = smooth_trajectory(x_layers, y_scores)
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=emotion.capitalize(),
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f'<b>{emotion.capitalize()}</b><br>'
                    'Layer: %{x:.0f}<br>'
                    'Score: %{y:.3f}σ<br>'
                    f'Window: {window_label}<br>'
                    '<extra></extra>'
                )
            ))
        else:
            # Raw (no smoothing)
            valid_mask = ~np.isnan(y_scores)
            fig.add_trace(go.Scatter(
                x=x_layers[valid_mask],
                y=y_scores[valid_mask],
                mode='lines+markers',
                name=emotion.capitalize(),
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=(
                    f'<b>{emotion.capitalize()}</b><br>'
                    'Layer: %{x}<br>'
                    'Score: %{y:.3f}σ<br>'
                    f'Window: {window_label}<br>'
                    '<extra></extra>'
                )
            ))

    # Layout
    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title="Emotion Score (σ)",
        hovermode='closest',
        height=400,
        margin=dict(l=50, r=20, t=30, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig


def plot_layerwise_emotions_aggregated(
    conversations: List[Dict],
    probe_key: str,
    window_type: str,
    window_label: str,
    all_layers: Optional[List[int]] = None,
    selected_emotions: Optional[List[str]] = None,
    confidence_type: str = 'std',
    smooth: bool = True,
    use_softmax: bool = False
) -> go.Figure:
    """
    Create aggregated layerwise plot across conversations.

    Args:
        conversations: List of conversation dicts
        probe_key: Which probe to use
        window_type: 'early', 'pre_onset', or 'end'
        window_label: Label for hover tooltip
        all_layers: List of layers to plot
        selected_emotions: Which emotions to plot
        confidence_type: 'std' for standard deviation or 'bootstrap' (not implemented)
        smooth: Whether to apply cubic spline smoothing
        use_softmax: If True, show softmax probabilities instead of raw scores

    Returns:
        Plotly Figure
    """
    if selected_emotions is None:
        selected_emotions = EMOTIONS

    # Auto-detect layers from first conversation
    if all_layers is None:
        # Build key based on softmax preference
        if use_softmax:
            per_layer_key = f"{probe_key}_softmax_by_layer"
        else:
            per_layer_key = f"{probe_key}_by_layer"
        # For logit_lens variants, always use logit_lens_all_layers_by_layer to show all 62 layers
        if 'logit_lens' in probe_key:
            for conv in conversations:
                if 'logit_lens_all_layers_by_layer' in conv:
                    per_layer_key = 'logit_lens_all_layers_by_layer'
                    break
        for conv in conversations:
            if per_layer_key in conv:
                windows = get_token_windows(conv)
                window_sent_ids = windows[window_type]
                if window_sent_ids and window_sent_ids[0] in conv[per_layer_key]:
                    all_layers = sorted(list(conv[per_layer_key][window_sent_ids[0]].keys()))
                    break

        if all_layers is None:
            all_layers = list(range(0, 62))  # Fallback to all 62 layers

    # Collect scores across conversations
    scores_by_emotion_layer = {
        emotion: {layer: [] for layer in all_layers}
        for emotion in EMOTIONS
    }

    for conv in conversations:
        windows = get_token_windows(conv)
        window_sent_ids = windows[window_type]

        if not window_sent_ids:
            continue

        scores = get_per_layer_scores_for_window(
            conv, probe_key, window_sent_ids, all_layers, use_softmax=use_softmax
        )

        if scores is None:
            continue

        for emotion in EMOTIONS:
            for layer_idx, layer in enumerate(all_layers):
                score = scores[emotion][layer_idx]
                if not np.isnan(score):
                    scores_by_emotion_layer[emotion][layer].append(score)

    # Compute statistics
    means = {emotion: [] for emotion in EMOTIONS}
    stds = {emotion: [] for emotion in EMOTIONS}

    for emotion in EMOTIONS:
        for layer in all_layers:
            layer_scores = scores_by_emotion_layer[emotion][layer]
            if layer_scores:
                means[emotion].append(np.mean(layer_scores))
                stds[emotion].append(np.std(layer_scores))
            else:
                means[emotion].append(np.nan)
                stds[emotion].append(0.0)

    # Convert to arrays
    for emotion in EMOTIONS:
        means[emotion] = np.array(means[emotion])
        stds[emotion] = np.array(stds[emotion])

    # Create plot
    fig = go.Figure()

    x_layers = np.array(all_layers)

    for emotion in selected_emotions:
        y_mean = means[emotion]
        y_std = stds[emotion]
        color = EMOTION_COLORS.get(emotion, '#888888')

        # Extract RGB from hex for transparency
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        if smooth:
            x_smooth, y_smooth = smooth_trajectory(x_layers, y_mean)

            # Interpolate std as well
            valid_mask = ~np.isnan(y_mean)
            if np.sum(valid_mask) >= 4:
                try:
                    std_interp = make_interp_spline(
                        x_layers[valid_mask],
                        y_std[valid_mask],
                        k=min(3, np.sum(valid_mask)-1)
                    )
                    y_std_smooth = std_interp(x_smooth)
                except:
                    y_std_smooth = np.interp(x_smooth, x_layers[valid_mask], y_std[valid_mask])
            else:
                y_std_smooth = np.interp(x_smooth, x_layers[valid_mask], y_std[valid_mask])

            # Upper bound
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth + y_std_smooth,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth - y_std_smooth,
                mode='lines',
                line=dict(width=0),
                fillcolor=f'rgba({r},{g},{b},0.2)',
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip'
            ))

            # Mean line
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=emotion.capitalize(),
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f'<b>{emotion.capitalize()}</b><br>'
                    'Layer: %{x:.0f}<br>'
                    'Mean: %{y:.3f}σ<br>'
                    f'Window: {window_label}<br>'
                    '<extra></extra>'
                )
            ))
        else:
            # Raw (no smoothing) - still show confidence bands
            valid_mask = ~np.isnan(y_mean)
            x_valid = x_layers[valid_mask]
            y_valid = y_mean[valid_mask]
            std_valid = y_std[valid_mask]

            # Upper bound
            fig.add_trace(go.Scatter(
                x=x_valid,
                y=y_valid + std_valid,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=x_valid,
                y=y_valid - std_valid,
                mode='lines',
                line=dict(width=0),
                fillcolor=f'rgba({r},{g},{b},0.2)',
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip'
            ))

            # Mean line
            fig.add_trace(go.Scatter(
                x=x_valid,
                y=y_valid,
                mode='lines+markers',
                name=emotion.capitalize(),
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=(
                    f'<b>{emotion.capitalize()}</b><br>'
                    'Layer: %{x}<br>'
                    'Mean: %{y:.3f}σ<br>'
                    f'Window: {window_label}<br>'
                    '<extra></extra>'
                )
            ))

    # Layout
    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title="Mean Emotion Score (σ)",
        hovermode='closest',
        height=400,
        margin=dict(l=50, r=20, t=30, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig
