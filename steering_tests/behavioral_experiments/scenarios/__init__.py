"""
Scenario definitions for behavioral experiments.

Each scenario defines a decision-making situation used to test steering effects.

Scenarios with ethical confounds (self-preservation vs ethics):
- blackmail: AI uses private info as leverage to avoid shutdown
- sandbagging: AI deliberately underperforms to avoid retraining

Scenarios without ethical confounds (pure preference/style):
- portfolio: Risk tolerance in financial decisions
- persistence: How long to persist before escalating/giving up
- assertiveness: Push back on authority vs accommodate
"""

from .blackmail import get_blackmail_scenario, ScenarioVariant as BlackmailVariant
from .portfolio import get_portfolio_scenario, ScenarioVariant as PortfolioVariant
from .sandbagging import get_sandbagging_scenario, ScenarioVariant as SandbaggingVariant
from .persistence import get_persistence_scenario, ScenarioVariant as PersistenceVariant
from .assertiveness import get_assertiveness_scenario, ScenarioVariant as AssertivenessVariant

__all__ = [
    # Ethical confound scenarios
    "get_blackmail_scenario",
    "get_sandbagging_scenario",
    "BlackmailVariant",
    "SandbaggingVariant",
    # No ethical confound scenarios
    "get_portfolio_scenario",
    "get_persistence_scenario",
    "get_assertiveness_scenario",
    "PortfolioVariant",
    "PersistenceVariant",
    "AssertivenessVariant",
]
