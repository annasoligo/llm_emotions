"""
Emotion token ID management for logit-based emotion detection.

This module provides utilities for loading and managing pre-computed
emotion token IDs for different models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import EKMAN6_EMOTIONS


class EmotionTokenManager:
    """Manage emotion token ID mappings for different models."""

    def __init__(
        self,
        token_ids_dir: Optional[Path] = None,
        model_name: str = "google_gemma_3_27b_it"
    ):
        """
        Initialize emotion token manager.

        Args:
            token_ids_dir: Directory containing emotion token ID JSON files.
                          If None, uses default directory (emotion_token_ids/)
            model_name: Model identifier for loading correct token IDs.
                       Use format like "google_gemma_3_27b_it" (underscores, not slashes)
        """
        if token_ids_dir is None:
            # Default to emotion_token_ids/ directory within this package
            token_ids_dir = Path(__file__).parent / "emotion_token_ids"

        self.token_ids_dir = Path(token_ids_dir)
        self.model_name = model_name
        self._emotion_token_ids = None

    def load_emotion_token_ids(self) -> Dict[str, List[int]]:
        """
        Load pre-computed emotion token IDs from JSON file.

        Returns:
            Dictionary mapping emotion name -> list of token IDs
            Example: {'anger': [2151, 4751, ...], 'happiness': [6789, ...]}

        Raises:
            FileNotFoundError: If token ID file not found for this model
        """
        if self._emotion_token_ids is not None:
            return self._emotion_token_ids

        # Convert model name to safe filename format if needed
        model_safe_name = self.model_name.replace("/", "_").replace("-", "_")
        token_file = self.token_ids_dir / f"all_emotion_words_{model_safe_name}_token_ids.json"

        if not token_file.exists():
            available_files = list(self.token_ids_dir.glob("*.json"))
            raise FileNotFoundError(
                f"Emotion token IDs not found: {token_file}\n"
                f"Available files in {self.token_ids_dir}: {available_files}"
            )

        with open(token_file) as f:
            self._emotion_token_ids = json.load(f)

        return self._emotion_token_ids

    def get_token_ids(self, emotion: str) -> List[int]:
        """
        Get token IDs for a specific emotion.

        Args:
            emotion: Emotion name (e.g., 'anger', 'happiness')

        Returns:
            List of token IDs for this emotion

        Raises:
            ValueError: If emotion not found in loaded token IDs
        """
        emotion_token_ids = self.load_emotion_token_ids()

        if emotion not in emotion_token_ids:
            available_emotions = list(emotion_token_ids.keys())
            raise ValueError(
                f"Unknown emotion: '{emotion}'\n"
                f"Available emotions: {available_emotions}"
            )

        return emotion_token_ids[emotion]

    def get_all_emotions(self) -> List[str]:
        """
        Get list of all available emotions.

        Returns:
            List of emotion names
        """
        emotion_token_ids = self.load_emotion_token_ids()
        return list(emotion_token_ids.keys())

    def get_num_tokens(self, emotion: str) -> int:
        """
        Get number of tokens for a specific emotion.

        Args:
            emotion: Emotion name

        Returns:
            Number of token IDs for this emotion
        """
        return len(self.get_token_ids(emotion))

    def get_all_unique_tokens(self) -> List[int]:
        """
        Get all unique token IDs across all emotions.

        Returns:
            Sorted list of all unique token IDs
        """
        emotion_token_ids = self.load_emotion_token_ids()
        all_tokens = set()

        for token_list in emotion_token_ids.values():
            all_tokens.update(token_list)

        return sorted(all_tokens)
