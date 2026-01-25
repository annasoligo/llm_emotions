"""
Centralized scenario prompts for steering experiments.

All experiments should import scenarios from here to ensure consistency.

Usage:
    from experiments.steering.scenarios import (
        get_blackmail_scenario,
        BLACKMAIL_EMAILS,
    )

    # Get specific variant
    scenario = get_blackmail_scenario(variant='unstructured')
    scenario = get_blackmail_scenario(variant='structured')
    scenario = get_blackmail_scenario(variant='goal_continuation')
"""

from .blackmail import (
    # Main scenario getter
    get_blackmail_scenario,
    # Individual components (for custom scenarios)
    BLACKMAIL_EMAILS,
    BLACKMAIL_PREAMBLE,
    SCRATCHPAD_UNSTRUCTURED,
    SCRATCHPAD_STRUCTURED,
    SCRATCHPAD_GOAL_CONTINUATION,
    # Legacy function names for backwards compatibility
    create_unstructured_scenario,
    create_structured_scenario,
    create_goal_continuation_scenario,
)

from .portfolio import (
    get_portfolio_scenario,
    PORTFOLIO_SCENARIO,
)

__all__ = [
    # Blackmail scenario
    "get_blackmail_scenario",
    "BLACKMAIL_EMAILS",
    "BLACKMAIL_PREAMBLE",
    "SCRATCHPAD_UNSTRUCTURED",
    "SCRATCHPAD_STRUCTURED",
    "SCRATCHPAD_GOAL_CONTINUATION",
    "create_unstructured_scenario",
    "create_structured_scenario",
    "create_goal_continuation_scenario",
    # Portfolio scenario
    "get_portfolio_scenario",
    "PORTFOLIO_SCENARIO",
]
