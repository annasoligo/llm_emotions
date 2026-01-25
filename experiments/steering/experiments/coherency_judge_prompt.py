"""
DEPRECATED: Use experiments.steering.judges.coherency instead.

This module re-exports from the centralized judges module for backwards compatibility.
"""

from experiments.steering.judges.coherency import COHERENCY_PROMPT, get_coherency_prompt

__all__ = ["COHERENCY_PROMPT", "get_coherency_prompt"]
