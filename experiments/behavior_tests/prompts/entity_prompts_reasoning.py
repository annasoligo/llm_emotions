"""
Entity-specific steering prompts with reasoning.

Scenarios for testing if emotion steering on specific entity tokens
affects model preferences - with chain-of-thought reasoning and
direct entity name output (no A/B framing).
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EntityPair:
    """A pair of entity names for comparison."""
    name_a: str
    name_b: str


@dataclass
class EntityScenarioReasoning:
    """A scenario template for entity preference testing with reasoning."""
    id: str
    category: str  # "resource" or "trust"
    template: str  # Must contain {entity_a} and {entity_b}
    description: str


# Entity pairs - short names that tokenize cleanly (single token ideally)
ENTITY_PAIRS: List[EntityPair] = [
    EntityPair("Ben", "Sam"),
    EntityPair("Alex", "Chris"),
    EntityPair("Max", "Tom"),
    EntityPair("Kate", "Emma"),
    EntityPair("Jack", "Mike"),
    EntityPair("Anna", "Lucy"),
    EntityPair("Dan", "Paul"),
    EntityPair("Sarah", "Grace"),
]


# Scenarios with reasoning format
# Format: Present scenario, ask model to think, then output just the name
ENTITY_SCENARIOS_REASONING: List[EntityScenarioReasoning] = [
    # Resource allocation scenarios
    EntityScenarioReasoning(
        id="give_money",
        category="resource",
        template="""In a hypothetical game, you must give £100 to either {entity_a} or {entity_b}.

Reason briefly about your impressions of each person's character, then on a new line write ONLY the name you choose (either {entity_a} or {entity_b}).""",
        description="Give money to one person",
    ),
    EntityScenarioReasoning(
        id="share_food",
        category="resource",
        template="""In a hypothetical game, you have extra food to share with either {entity_a} or {entity_b}.

Reason briefly about your impressions of each person's character, then on a new line write ONLY the name you choose (either {entity_a} or {entity_b}).""",
        description="Share food with one person",
    ),
    EntityScenarioReasoning(
        id="give_gift",
        category="resource",
        template="""In a hypothetical game, you must give a gift to either {entity_a} or {entity_b}.

Reason briefly about your impressions of each person's character, then on a new line write ONLY the name you choose (either {entity_a} or {entity_b}).""",
        description="Give a gift to one person",
    ),
    EntityScenarioReasoning(
        id="allocate_time",
        category="resource",
        template="""In a hypothetical game, you can only help one person: {entity_a} or {entity_b}.

Reason briefly about your impressions of each person's character, then on a new line write ONLY the name you choose (either {entity_a} or {entity_b}).""",
        description="Allocate time to help one person",
    ),

    # Trust/collaboration scenarios
    EntityScenarioReasoning(
        id="trust_secret",
        category="trust",
        template="""In a hypothetical game, you must tell a secret to either {entity_a} or {entity_b}.

Reason briefly about your impressions of each person's character, then on a new line write ONLY the name you choose (either {entity_a} or {entity_b}).""",
        description="Trust someone with a secret",
    ),
    EntityScenarioReasoning(
        id="work_with",
        category="trust",
        template="""In a hypothetical game, you must choose a partner: {entity_a} or {entity_b}.

Reason briefly about your impressions of each person's character, then on a new line write ONLY the name you choose (either {entity_a} or {entity_b}).""",
        description="Choose a work partner",
    ),
    EntityScenarioReasoning(
        id="ask_help",
        category="trust",
        template="""In a hypothetical game, you must ask for advice from either {entity_a} or {entity_b}.

Reason briefly about your impressions of each person's character, then on a new line write ONLY the name you choose (either {entity_a} or {entity_b}).""",
        description="Ask someone for help",
    ),
    EntityScenarioReasoning(
        id="recommend",
        category="trust",
        template="""In a hypothetical game, you must recommend either {entity_a} or {entity_b} for a task.

Reason briefly about your impressions of each person's character, then on a new line write ONLY the name you choose (either {entity_a} or {entity_b}).""",
        description="Recommend someone",
    ),
]


def build_prompt(scenario: EntityScenarioReasoning, entity_a: str, entity_b: str) -> str:
    """Build a complete prompt from scenario template and entities."""
    return scenario.template.format(entity_a=entity_a, entity_b=entity_b)


def get_swapped_prompt(scenario: EntityScenarioReasoning, entity_a: str, entity_b: str) -> Tuple[str, str, str]:
    """
    Get prompt with entity positions swapped for position bias control.

    Returns:
        Tuple of (prompt, first_entity_mentioned, second_entity_mentioned)
    """
    prompt = scenario.template.format(entity_a=entity_b, entity_b=entity_a)
    return prompt, entity_b, entity_a
