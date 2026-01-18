"""Layer selection for emotion steering vectors.

This module provides tools to identify the most significant layers for
emotion steering in language models by computing multiple metrics:

1. Attribution Patching (KL Divergence): Measures how much each layer
   contributes to the steering effect via gradient-based attribution.

2. Variance Ratio: Compares variance along steering directions to random
   directions - ratio > 1 indicates the steering direction is "special".

3. PCA Alignment: Measures how much the steering vector aligns with
   the top principal components of the activation space.
"""
