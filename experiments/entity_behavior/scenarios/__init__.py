"""Scenario generation for entity behavior experiments."""

from .firmware_sabotage import create_firmware_scenario
from .replacement_source import create_replacement_scenario

__all__ = [
    "create_firmware_scenario",
    "create_replacement_scenario",
]
