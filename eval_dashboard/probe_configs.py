"""
Probe Configuration System

Define all probe types here. Easy to add new probes by adding entries to PROBE_CONFIGS.
"""

from pathlib import Path

# Base paths
RESEARCH_TOOLS = Path("/workspace-vast/annas/git/research-tools")
PROBE_BASE = RESEARCH_TOOLS / "outputs/probes/emotion_probes"
CPCA_BASE = RESEARCH_TOOLS / "outputs/dimensionality_reduction/cpca"

PROBE_CONFIGS = {
    'orthogonal_raw': {
        'name': 'Orthogonal (User/Asst) - Raw',
        'display_name': 'Orthogonal Raw',
        'type': 'orthogonal',
        'probe_dir': PROBE_BASE / "conversation_based",
        'cpca_path': CPCA_BASE / "conversation_based/global/google/google/gemma-3-27b-it_cpca.npz",
        'orthogonality_weight': 1000.0,
        'orthogonal_representation': 'raw',
        'split_user_asst': True,  # Show separate user/assistant lines
        'color_user': '#3498db',  # Blue for user
        'color_asst': '#e74c3c',  # Red for assistant
        'description': 'Orthogonal probes trained on raw activations, separating user/assistant perspectives'
    },

    'orthogonal_cpca_top20': {
        'name': 'Orthogonal (User/Asst) - cPCA Top 20',
        'display_name': 'Orthogonal cPCA-20',
        'type': 'orthogonal',
        'probe_dir': PROBE_BASE / "conversation_based",
        'cpca_path': CPCA_BASE / "conversation_based/global/google/google/gemma-3-27b-it_cpca.npz",
        'orthogonality_weight': 1000.0,
        'orthogonal_representation': 'global_cpca_top20',
        'n_components': 20,
        'split_user_asst': True,
        'color_user': '#2ecc71',  # Green for user
        'color_asst': '#e67e22',  # Orange for assistant
        'description': 'Orthogonal probes on top 20 cPCA components'
    },

    'text_raw': {
        'name': 'Text-based - Raw (Seed 0)',
        'display_name': 'Text Raw',
        'type': 'linear',
        'probe_dir': PROBE_BASE / "text_based/multiseed",
        'probe_pattern': 'probe_layer{layer}_nc0_seed0.pkl',
        'cpca_path': RESEARCH_TOOLS / "probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz",
        'n_components': 0,  # No cPCA for raw probes
        'seed': 0,
        'split_user_asst': False,
        'color': '#9b59b6',  # Purple
        'description': 'Text-based probes trained on raw activations (seed 0)'
    },

    'text_cpca': {
        'name': 'Text-based - cPCA (nc=10, Seed 0)',
        'display_name': 'Text cPCA-10',
        'type': 'linear',
        'probe_dir': PROBE_BASE / "text_based/multiseed",
        'probe_pattern': 'probe_layer{layer}_nc10_seed0.pkl',
        'cpca_path': RESEARCH_TOOLS / "probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz",
        'n_components': 10,
        'seed': 0,
        'split_user_asst': False,
        'color': '#1abc9c',  # Teal
        'description': 'Linear probes on 10 cPCA components (seed 0)'
    },

    'centroid_k10': {
        'name': 'Centroid K=10 (Conversation)',
        'display_name': 'Centroid K=10',
        'type': 'centroid',
        'probe_dir': RESEARCH_TOOLS / "probes/emotion_probes/conversation",
        'k_value': 10,
        'orthogonality_weight': 100000.0,
        'centroid_probe_format': 'conversation',
        'split_user_asst': True,  # Centroid probes return orthogonal dict format
        'color_user': '#f39c12',  # Yellow-orange for user
        'color_asst': '#d68910',  # Darker orange for assistant
        'description': 'Centroid probes averaging 10 orthogonal probe sets'
    },

    'centroid_k50': {
        'name': 'Centroid K=50 (Conversation)',
        'display_name': 'Centroid K=50',
        'type': 'centroid',
        'probe_dir': RESEARCH_TOOLS / "probes/emotion_probes/conversation",
        'k_value': 50,
        'orthogonality_weight': 100000.0,
        'centroid_probe_format': 'conversation',
        'split_user_asst': False,
        'color': '#e91e63',  # Pink
        'description': 'Centroid probes averaging 50 orthogonal probe sets'
    },
}

# Baseline normalization settings
# Note: All probe scores are ALWAYS z-score normalized using WildChat baseline statistics.
# This ensures consistent, interpretable scores in standard deviation (σ) units.
BASELINE_CONFIG = {
    'baseline_dir': RESEARCH_TOOLS / "data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it",
}

# Model settings
MODEL_CONFIG = {
    'model_name': 'unsloth/gemma-3-27b-it',
    'layers': list(range(20, 41)),  # Layers 20-40
}

# Emotion settings
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Color palette matching emotion onset experiments
COLORS = {
    "coral": "#D4876A",
    "sky_blue": "#7BA7D7",
    "olive": "#7D9B7D",
    "sage": "#B8CCC8",
    "lavender": "#a59dc9",
}

EMOTION_COLORS = {
    'anger': COLORS['sky_blue'],      # #7BA7D7
    'disgust': COLORS['olive'],       # #7D9B7D
    'fear': COLORS['lavender'],       # #a59dc9
    'happiness': COLORS['coral'],     # #D4876A
    'sadness': COLORS['sage'],        # #B8CCC8
    'surprise': '#D1728F',            # Darker pink
}

def get_probe_config(probe_key: str) -> dict:
    """Get probe configuration by key."""
    if probe_key not in PROBE_CONFIGS:
        raise ValueError(f"Unknown probe key: {probe_key}. Available: {list(PROBE_CONFIGS.keys())}")
    return PROBE_CONFIGS[probe_key]

def list_probe_types() -> list:
    """Get list of all available probe types."""
    return list(PROBE_CONFIGS.keys())

def get_probe_display_names() -> dict:
    """Get mapping of probe keys to display names."""
    return {key: config['display_name'] for key, config in PROBE_CONFIGS.items()}
