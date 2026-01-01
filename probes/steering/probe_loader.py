"""Load emotion probes and create steering vector builders."""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from .vectors import ProbeSteeringVectorBuilder


def load_emotion_probe(
    probe_path: Path,
    layer: int,
    probe_type: str = 'assistant',
    cpca_path: Optional[Path] = None
) -> ProbeSteeringVectorBuilder:
    """Load emotion probe and create steering vector builder.

    Args:
        probe_path: Path to probe pickle file
        layer: Layer index for steering
        probe_type: 'user' or 'assistant'
        cpca_path: Optional path to cPCA components NPZ file

    Returns:
        ProbeSteeringVectorBuilder instance
    """
    probe_path = Path(probe_path)

    with open(probe_path, 'rb') as f:
        data = pickle.load(f)

    # Extract model and labels
    model = data['model']
    emotion_labels = data['label_names']

    # Convert PyTorch weights to numpy
    if isinstance(model.weight, torch.Tensor):
        probe_weights = model.weight.detach().cpu().numpy()
    else:
        probe_weights = np.array(model.weight)

    # Load cPCA if provided
    cpcs = None
    if cpca_path:
        cpca_data = np.load(cpca_path)
        cpcs = cpca_data['components']

    return ProbeSteeringVectorBuilder(
        probe_weights=probe_weights,
        emotion_labels=emotion_labels,
        cpcs=cpcs,
        layer=layer,
        probe_type=probe_type
    )
