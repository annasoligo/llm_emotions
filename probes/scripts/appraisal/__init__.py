"""Appraisal minimal-pair data generation pipeline.

This module provides an end-to-end pipeline for:
1. Generating manipulation cards from axis hypotheses
2. Auditing cards for quality
3. Generating minimal-pair scenarios
4. Auditing scenarios
5. Creating paraphrases for invariance testing
6. Logging activations from target models
"""

from .config import AppraisalConfig
from .pipeline import AppraisalPipeline

__all__ = ["AppraisalConfig", "AppraisalPipeline"]
