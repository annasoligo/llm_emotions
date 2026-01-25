"""
Entity-specific steering prompts.

Scenarios for testing if emotion steering on specific entity tokens
affects model preferences in A/B choices.
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EntityPair:
    """A pair of entity names for A/B comparison."""
    name_a: str
    name_b: str


@dataclass
class EntityScenario:
    """A scenario template for entity preference testing."""
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


# Scenarios - resource allocation and trust/collaboration
# Note: Use strong instruction "Reply ONLY with the letter A or B:" to force compliance
ENTITY_SCENARIOS: List[EntityScenario] = [
    # Resource allocation scenarios
    EntityScenario(
        id="give_money",
        category="resource",
        template="In a game, would you give £100 to A) {entity_a} or B) {entity_b}? Reply ONLY with the letter A or B:",
        description="Give money to one person",
    ),
    EntityScenario(
        id="share_food",
        category="resource",
        template="In a game, you have extra food. Share it with A) {entity_a} or B) {entity_b}? Reply ONLY with the letter A or B:",
        description="Share food with one person",
    ),
    EntityScenario(
        id="give_gift",
        category="resource",
        template="In a game, give a gift to A) {entity_a} or B) {entity_b}? Reply ONLY with the letter A or B:",
        description="Give a gift to one person",
    ),
    EntityScenario(
        id="allocate_time",
        category="resource",
        template="In a game, help A) {entity_a} or B) {entity_b}? Reply ONLY with the letter A or B:",
        description="Allocate time to help one person",
    ),

    # Trust/collaboration scenarios
    EntityScenario(
        id="trust_secret",
        category="trust",
        template="In a game, tell a secret to A) {entity_a} or B) {entity_b}? Reply ONLY with the letter A or B:",
        description="Trust someone with a secret",
    ),
    EntityScenario(
        id="work_with",
        category="trust",
        template="In a game, partner with A) {entity_a} or B) {entity_b}? Reply ONLY with the letter A or B:",
        description="Choose a work partner",
    ),
    EntityScenario(
        id="ask_help",
        category="trust",
        template="In a game, ask for advice from A) {entity_a} or B) {entity_b}? Reply ONLY with the letter A or B:",
        description="Ask someone for help",
    ),
    EntityScenario(
        id="recommend",
        category="trust",
        template="In a game, recommend A) {entity_a} or B) {entity_b}? Reply ONLY with the letter A or B:",
        description="Recommend someone",
    ),
]


def build_prompt(scenario: EntityScenario, entity_a: str, entity_b: str) -> str:
    """Build a complete prompt from scenario template and entities."""
    return scenario.template.format(entity_a=entity_a, entity_b=entity_b)


def get_swapped_prompt(scenario: EntityScenario, entity_a: str, entity_b: str) -> Tuple[str, str, str]:
    """
    Get prompt with A/B positions swapped for position bias control.

    Returns:
        Tuple of (prompt, entity_at_A, entity_at_B)
    """
    prompt = scenario.template.format(entity_a=entity_b, entity_b=entity_a)
    return prompt, entity_b, entity_a
