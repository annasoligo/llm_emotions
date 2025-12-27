"""Steering vector construction from PCs and probes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Dict
import numpy as np
import json


@dataclass
class SteeringVector:
    """Steering vector for activation intervention.

    Attributes:
        vector: Steering direction in activation space [hidden_dim]
        layer: Which transformer layer to apply at
        name: Human-readable identifier
        source: Origin type ("pc", "probe", "cluster_centroid")
        metadata: Additional info (emotion, scale, etc)
    """
    vector: np.ndarray
    layer: int
    name: str
    source: str
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate steering vector."""
        if self.vector.ndim != 1:
            raise ValueError(f"vector must be 1D, got shape {self.vector.shape}")

        if np.isnan(self.vector).any() or np.isinf(self.vector).any():
            raise ValueError("vector contains NaN or Inf values")

        if self.layer < 0:
            raise ValueError(f"layer must be non-negative, got {self.layer}")

    def normalize(self) -> "SteeringVector":
        """Return normalized copy (unit length)."""
        norm = np.linalg.norm(self.vector)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")

        return SteeringVector(
            vector=self.vector / norm,
            layer=self.layer,
            name=self.name,
            source=self.source,
            metadata=self.metadata.copy(),
        )

    def scale(self, factor: float) -> "SteeringVector":
        """Return scaled copy."""
        return SteeringVector(
            vector=self.vector * factor,
            layer=self.layer,
            name=f"{self.name}_x{factor}",
            source=self.source,
            metadata={**self.metadata, "scale_factor": factor},
        )

    def save(self, path: Path) -> None:
        """Save to NPZ + JSON files.

        Args:
            path: Output path (e.g., 'steering_vec.npz')

        Raises:
            OSError: If cannot write files
        """
        path = Path(path)

        # Save vector as NPZ
        try:
            np.savez(path, vector=self.vector, layer=self.layer)
        except OSError as e:
            raise OSError(f"Failed to save vector to {path}: {e}")

        # Save metadata as JSON
        meta_path = path.with_suffix('.json')
        metadata_dict = {
            "name": self.name,
            "source": self.source,
            "layer": int(self.layer),
            "vector_shape": list(self.vector.shape),
            "vector_norm": float(np.linalg.norm(self.vector)),
            "metadata": self.metadata,
        }

        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        except OSError as e:
            raise OSError(f"Failed to save metadata to {meta_path}: {e}")

    @classmethod
    def load(cls, path: Path) -> "SteeringVector":
        """Load from NPZ + JSON files.

        Args:
            path: Input path (e.g., 'steering_vec.npz')

        Returns:
            SteeringVector instance

        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If data invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Vector file not found: {path}")

        # Load vector
        try:
            data = np.load(path)
            vector = data["vector"]
            layer = int(data["layer"])
        except Exception as e:
            raise ValueError(f"Failed to load vector from {path}: {e}")

        # Load metadata
        meta_path = path.with_suffix('.json')
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        try:
            with open(meta_path, 'r') as f:
                meta_dict = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load metadata from {meta_path}: {e}")

        return cls(
            vector=vector,
            layer=layer,
            name=meta_dict["name"],
            source=meta_dict["source"],
            metadata=meta_dict.get("metadata", {}),
        )


class PCSteeringVectorBuilder:
    """Build steering vectors from cPCA components."""

    def __init__(self, cpcs: np.ndarray, layer: int):
        """Initialize builder.

        Args:
            cpcs: cPCA components [hidden_dim, n_components]
            layer: Layer index these components are from

        Raises:
            ValueError: If cpcs has wrong shape or contains NaN/Inf
        """
        if cpcs.ndim != 2:
            raise ValueError(f"cpcs must be 2D, got shape {cpcs.shape}")

        if np.isnan(cpcs).any() or np.isinf(cpcs).any():
            raise ValueError("cpcs contains NaN or Inf values")

        self.cpcs = cpcs
        self.layer = layer
        self.hidden_dim, self.n_components = cpcs.shape

    def get_pc_vector(
        self,
        pc_idx: int,
        scale: float = 1.0,
        normalize: bool = True,
    ) -> SteeringVector:
        """Get steering vector from single PC.

        Args:
            pc_idx: PC index (0 to n_components-1)
            scale: Scaling factor
            normalize: Whether to normalize to unit length before scaling

        Returns:
            SteeringVector

        Raises:
            ValueError: If pc_idx out of range
        """
        if pc_idx < 0 or pc_idx >= self.n_components:
            raise ValueError(
                f"pc_idx {pc_idx} out of range [0, {self.n_components})"
            )

        vector = self.cpcs[:, pc_idx].copy()

        if normalize:
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError(f"PC {pc_idx} is zero vector")
            vector = vector / norm

        vector = vector * scale

        return SteeringVector(
            vector=vector,
            layer=self.layer,
            name=f"PC{pc_idx}_L{self.layer}",
            source="pc",
            metadata={"pc_idx": pc_idx, "scale": scale, "normalized": normalize},
        )

    def get_multiple_pcs(
        self,
        pc_indices: List[int],
        scales: Optional[List[float]] = None,
        normalize: bool = True,
    ) -> SteeringVector:
        """Get steering vector from weighted sum of multiple PCs.

        Args:
            pc_indices: List of PC indices
            scales: Optional scaling factors per PC (default: all 1.0)
            normalize: Whether to normalize each PC before combining

        Returns:
            SteeringVector (combined)

        Raises:
            ValueError: If indices invalid or lists mismatched
        """
        if not pc_indices:
            raise ValueError("pc_indices cannot be empty")

        if scales is None:
            scales = [1.0] * len(pc_indices)

        if len(scales) != len(pc_indices):
            raise ValueError(
                f"scales length {len(scales)} != pc_indices length {len(pc_indices)}"
            )

        # Validate all indices
        for idx in pc_indices:
            if idx < 0 or idx >= self.n_components:
                raise ValueError(
                    f"pc_idx {idx} out of range [0, {self.n_components})"
                )

        # Combine PCs
        combined = np.zeros(self.hidden_dim)
        for pc_idx, scale in zip(pc_indices, scales):
            vec = self.cpcs[:, pc_idx].copy()

            if normalize:
                norm = np.linalg.norm(vec)
                if norm == 0:
                    raise ValueError(f"PC {pc_idx} is zero vector")
                vec = vec / norm

            combined += vec * scale

        name = f"PCs{pc_indices}_L{self.layer}"

        return SteeringVector(
            vector=combined,
            layer=self.layer,
            name=name,
            source="pc",
            metadata={
                "pc_indices": pc_indices,
                "scales": scales,
                "normalized": normalize,
            },
        )


class ProbeSteeringVectorBuilder:
    """Build steering vectors from probe directions."""

    # Emotion aliases
    EMOTION_ALIASES = {
        "joy": "happiness",
        "happy": "happiness",
        "sad": "sadness",
        "angry": "anger",
        "scared": "fear",
        "surprised": "surprise",
        "disgusted": "disgust",
    }

    def __init__(
        self,
        probe_weights: np.ndarray,
        emotion_labels: List[str],
        cpcs: Optional[np.ndarray] = None,
        layer: int = 0,
        probe_type: str = "user",
    ):
        """Initialize builder.

        Args:
            probe_weights: Probe coefficients [n_emotions, n_features]
            emotion_labels: Emotion names corresponding to rows
            cpcs: Optional cPCA components [hidden_dim, n_components] for projection
            layer: Layer index
            probe_type: "user" or "assistant"

        Raises:
            ValueError: If shapes invalid or contains NaN/Inf
        """
        if probe_weights.ndim != 2:
            raise ValueError(f"probe_weights must be 2D, got shape {probe_weights.shape}")

        if len(emotion_labels) != probe_weights.shape[0]:
            raise ValueError(
                f"emotion_labels length {len(emotion_labels)} != "
                f"probe_weights rows {probe_weights.shape[0]}"
            )

        if np.isnan(probe_weights).any() or np.isinf(probe_weights).any():
            raise ValueError("probe_weights contains NaN or Inf")

        if cpcs is not None:
            if cpcs.ndim != 2:
                raise ValueError(f"cpcs must be 2D, got shape {cpcs.shape}")
            if cpcs.shape[1] != probe_weights.shape[1]:
                raise ValueError(
                    f"cpcs components {cpcs.shape[1]} != "
                    f"probe features {probe_weights.shape[1]}"
                )

        self.probe_weights = probe_weights
        self.emotion_labels = [e.lower() for e in emotion_labels]
        self.cpcs = cpcs
        self.layer = layer
        self.probe_type = probe_type

    def _normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion name using aliases."""
        emotion = emotion.lower()
        return self.EMOTION_ALIASES.get(emotion, emotion)

    def get_emotion_vector(
        self,
        emotion: str,
        scale: float = 1.0,
        normalize: bool = True,
    ) -> SteeringVector:
        """Get steering vector for single emotion.

        Args:
            emotion: Emotion name (e.g., "happiness", "anger")
            scale: Scaling factor
            normalize: Whether to normalize to unit length before scaling

        Returns:
            SteeringVector

        Raises:
            ValueError: If emotion not found
        """
        emotion = self._normalize_emotion(emotion)

        if emotion not in self.emotion_labels:
            raise ValueError(
                f"Emotion '{emotion}' not found. Available: {self.emotion_labels}"
            )

        emotion_idx = self.emotion_labels.index(emotion)
        weights = self.probe_weights[emotion_idx]

        # Project to activation space if cPCs provided
        if self.cpcs is not None:
            vector = self.cpcs @ weights
        else:
            vector = weights.copy()

        if normalize:
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError(f"Emotion '{emotion}' has zero vector")
            vector = vector / norm

        vector = vector * scale

        return SteeringVector(
            vector=vector,
            layer=self.layer,
            name=f"{self.probe_type}_{emotion}_L{self.layer}",
            source=f"probe_{self.probe_type}",
            metadata={
                "emotion": emotion,
                "scale": scale,
                "normalized": normalize,
                "probe_type": self.probe_type,
            },
        )

    def get_contrast_vector(
        self,
        emotion_pos: str,
        emotion_neg: str,
        scale_pos: float = 1.0,
        scale_neg: float = 1.0,
        normalize: bool = True,
    ) -> SteeringVector:
        """Get contrast steering vector (emotion_pos - emotion_neg).

        Args:
            emotion_pos: Positive emotion to steer toward
            emotion_neg: Negative emotion to steer away from
            scale_pos: Scale for positive emotion
            scale_neg: Scale for negative emotion
            normalize: Whether to normalize final vector

        Returns:
            SteeringVector (contrast)

        Raises:
            ValueError: If emotions not found
        """
        # Get individual vectors (without final normalization)
        vec_pos = self.get_emotion_vector(emotion_pos, scale=scale_pos, normalize=False)
        vec_neg = self.get_emotion_vector(emotion_neg, scale=scale_neg, normalize=False)

        # Compute contrast
        contrast = vec_pos.vector - vec_neg.vector

        if normalize:
            norm = np.linalg.norm(contrast)
            if norm == 0:
                raise ValueError("Contrast vector is zero")
            contrast = contrast / norm

        return SteeringVector(
            vector=contrast,
            layer=self.layer,
            name=f"{self.probe_type}_{emotion_pos}_vs_{emotion_neg}_L{self.layer}",
            source=f"probe_{self.probe_type}_contrast",
            metadata={
                "emotion_pos": emotion_pos,
                "emotion_neg": emotion_neg,
                "scale_pos": scale_pos,
                "scale_neg": scale_neg,
                "normalized": normalize,
                "probe_type": self.probe_type,
            },
        )
