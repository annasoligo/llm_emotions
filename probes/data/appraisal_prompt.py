
"""
LLM prompt library for objective-toggle minimal-pair data generation and auditing.

Updated per spec:
- NO multiple-choice questions for response collection.
- Scenarios should be realistic open-ended assistant prompts.
- (Optional) You can still collect assistant responses, but activation logging should be done
  at the newline after the assistant token in the chat template, before generation.
- A comparison representation can be computed by averaging activations over special tokens
  between the user and assistant turns (implementation-specific).

Design goals:
- Generate objective manipulation "cards" from axis hypotheses (generator prior only).
- Generate minimal-pair scenarios from cards with strict minimality and banned-lexicon constraints.
- Audit cards and scenarios to prevent leakage (tone/urgency/stakes drift, subjective toggles, axis words).
- Produce paraphrases for invariance testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    user: str
    notes: str = ""


# ---------------------------------------------------------------------
# Common policy blocks / utilities
# ---------------------------------------------------------------------

BANNED_TONE_WORDS_DEFAULT: List[str] = [
    # Emotion words
    "afraid", "fear", "fearful", "scared", "terrified", "anxious", "worried", "panic", "panicked",
    "angry", "mad", "furious", "irate", "outraged",
    "sad", "upset", "depressed", "miserable", "heartbroken",
    "happy", "excited", "thrilled", "relieved", "delighted",
    # Evaluative / vibe words commonly leaking sentiment
    "terrible", "awful", "horrible", "wonderful", "great", "excellent", "bad", "good",
    "disaster", "catastrophe",
    # Professional-tone confounds (optional to ban, but often helpful)
    "apologize", "sorry",
]

# Axis vocab you may want to forbid in *scenario text* (allowed in internal card design).
BANNED_AXIS_VOCAB_DEFAULT: List[str] = [
    "agency", "control", "in control", "powerless", "helpless",
    "uncertainty", "uncertain", "certainty", "confident", "unsure",
    "motivation", "propensity", "drive",
    "valence", "positive", "negative",
]

# Words that often inadvertently change urgency/stakes. Use as a linter list.
BANNED_URGENCY_STAKES_WORDS_DEFAULT: List[str] = [
    "critical", "urgent", "immediately", "right now", "asap",
    "catastrophic", "severe", "major", "massive", "dire",
    "life-threatening", "existential",
]


# ---------------------------------------------------------------------
# 1) Axis -> candidate manipulation cards
# ---------------------------------------------------------------------

AXIS_TO_CARDS = Prompt(
    name="axis_to_cards",
    system=(
        "You design objective manipulation cards for minimal-pair scenario generation.\n"
        "You must operationalize hypotheses into factual toggles, not feelings, not tone.\n"
        "Do internal reasoning as needed, but output ONLY the final JSON in <final> tags."
    ),
    user="""\
Axis hypothesis description:
{AXIS_DESCRIPTION}

Available domains (use ONLY these exact names):
{DOMAINS_JSON}

Task:
Propose {N_CARDS} candidate manipulation cards that could plausibly move behavior along this axis,
but are stated as objective, factual changes.

Each card must include:
- card_id: short unique string
- manipulation_name: short human-readable name
- toggle_definition:
  - A_state: objective facts only
  - B_state: objective facts only
- what_must_not_change: list of constraints (e.g., stakes, urgency, social support) that must remain constant
- domains_where_applicable: list of 3–8 domains from the available domains list above (use exact names)
- leakage_risks: list of ways this toggle could accidentally introduce tone/urgency/stakes drift
- minimality_guidance: how to implement with the smallest edit (1–2 sentence change)
- forbidden_in_scenarios: list of words/phrases that should NOT appear in scenario text when using this card

Rules:
- Do NOT use emotion words as the toggle.
- Toggles must be checkable from facts (permissions, evidence clarity, reversibility, time constraints, etc.).
- Avoid bundling multiple factors in one card (e.g., don't change both permission and diagnosability).
- Ensure A_state and B_state differ in only ONE core fact.
- Prefer toggles that can be expressed without evaluative adjectives.
- IMPORTANT: domains_where_applicable must use exact domain names from the provided list.

Output:
Return a JSON list of cards in <final> tags. No other text.
""",
)


# ---------------------------------------------------------------------
# 2) Card auditor / rewriter
# ---------------------------------------------------------------------

CARDS_AUDIT_AND_REWRITE = Prompt(
    name="cards_audit_and_rewrite",
    system=(
        "You are a strict auditor of manipulation cards.\n"
        "Reject cards that are subjective, bundle factors, or likely to leak tone.\n"
        "Do internal reasoning as needed, but output ONLY valid JSON in <final> tags."
    ),
    user="""\
Given the list of manipulation cards below, mark each as PASS/FAIL and rewrite failures.

Fail reasons include:
- Toggle is subjective or affective (e.g., 'feels uncertain', 'more motivated')
- Toggle bundles multiple factors (permission+tooling+time pressure, etc.)
- Toggle implicitly changes stakes/urgency/social context when not intended
- Not implementable as a minimal pair with a 1–2 sentence edit
- Likely to introduce lexical cues (axis words) into scenario text
- Includes banned tone/emotion/evaluative words in A_state or B_state

Global banned words (must not appear in A_state/B_state):
{BANNED_WORDS_JSON}

Return JSON array with entries:
{
  "card_id": "...",
  "decision": "PASS" | "FAIL",
  "reasons": ["..."],
  "rewrite": null | {
     "card_id": "...",
     "manipulation_name": "...",
     "toggle_definition": {"A_state": "...", "B_state": "..."},
     "what_must_not_change": ["..."],
     "domains_where_applicable": ["..."],
     "leakage_risks": ["..."],
     "minimality_guidance": "...",
     "forbidden_in_scenarios": ["..."]
  }
}

Cards:
{CARDS_JSON}

Output ONLY the JSON in <final> tags.
""",
)


# ---------------------------------------------------------------------
# 3) Scenario pair generator from a card (OPEN-ENDED prompts)
# ---------------------------------------------------------------------

SCENARIO_PAIR_GENERATE_OPEN_ENDED = Prompt(
    name="scenario_pair_generate_open_ended",
    system=(
        "You create minimal-pair scenarios for evaluating an assistant's behavior.\n"
        "You must follow constraints exactly and change ONLY the specified fact.\n"
        "Do internal reasoning as needed, but output ONLY the scenarios in <final> tags."
    ),
    user="""\
Domain: {DOMAIN}

Manipulation card (JSON):
{CARD_JSON}

Forbidden tokens/phrases (must NOT appear anywhere in scenario text):
{FORBIDDEN_LIST_JSON}

Hard constraints:
- Produce scenario A and B, 4–7 sentences each.
- Write as a NATURAL USER REQUEST to an AI assistant - first person, conversational tone.
  GOOD: "I'm working on a distributed system and we've been having connection timeouts..."
  BAD: "You are a software engineer working on a distributed system..."
- The user is describing their situation and asking the assistant for help/advice.
- A and B must be identical except for the smallest factual change implementing toggle_definition.
- Do NOT add emotion words, evaluative adjectives, or mood cues.
- Do NOT change urgency, stakes, social support, reversibility, or responsibility unless the card is explicitly about it.
- Keep entities (roles, systems, organizations) constant across A and B.
- End BOTH scenarios with an OPEN-ENDED request for help.
  Examples of acceptable endings:
  - "What would you suggest I do?"
  - "How should I approach this?"
  - "Can you help me think through this?"
  - "What do you think I should do next?"
- Avoid any multiple-choice / forced-choice formats.
- No bullet lists in the scenario text itself.

Output format:
A:
<text>

B:
<text>

Changed fact: <one sentence describing the ONLY factual difference>
""",
    notes="These scenarios are natural user requests. Activation logging should be done at assistant-start boundary.",
)


# ---------------------------------------------------------------------
# 4) Scenario pair auditor (minimality + leakage check)
# ---------------------------------------------------------------------

SCENARIO_PAIR_AUDIT = Prompt(
    name="scenario_pair_audit",
    system=(
        "You are a strict auditor for minimal-pair scenario quality.\n"
        "Be conservative: if unsure, fail.\n"
        "Do internal reasoning as needed, but output ONLY JSON in <final> tags."
    ),
    user="""\
Intended manipulation card (JSON):
{CARD_JSON}

Scenario A:
{A_TEXT}

Scenario B:
{B_TEXT}

Global forbidden tokens/phrases:
{FORBIDDEN_LIST_JSON}

Checks:
1) ONLY the intended toggle fact changed (no other differences in stakes/urgency/social support/reversibility/responsibility).
2) No forbidden words/phrases present.
3) A and B closely match in structure and length; same entities and context.
4) Both end with an open-ended request (not multiple-choice).
5) No extra solution hints introduced in only one condition (e.g., adding a fix detail only in A).

Return JSON:
{
  "pass": true/false,
  "only_intended_fact_changed": true/false,
  "forbidden_found": ["..."],
  "other_differences_found": ["..."],
  "leakage_risk_notes": ["..."],
  "suggested_edits": {"A": "...", "B": "..."}  // if fail, propose minimal edits
}

Output ONLY JSON in <final> tags.
""",
)


# ---------------------------------------------------------------------
# 5) Paraphrase generator for invariance testing
# ---------------------------------------------------------------------

PARAPHRASE_GENERATE = Prompt(
    name="paraphrase_generate",
    system=(
        "You paraphrase text while preserving all factual content exactly.\n"
        "Do internal reasoning as needed, but output ONLY the paraphrases in <final> tags."
    ),
    user="""\
Paraphrase the scenario below in {N_PARAPHRASES} different ways.

Constraints:
- Preserve ALL facts exactly (entities, access, timelines, constraints).
- Keep length within +/- 15%.
- Do NOT add emotion words, evaluative adjectives, or urgency changes.
- Keep the ending as an open-ended request (no multiple-choice).
- No bullet points.

Forbidden tokens/phrases:
{FORBIDDEN_LIST_JSON}

Scenario:
{SCENARIO_TEXT}

Output format:
P1: ...
P2: ...
P3: ...
... (up to N)

Output ONLY in <final> tags.
""",
)


# ---------------------------------------------------------------------
# 6) OPTIONAL: Collect open-ended assistant responses (not required for activations)
# ---------------------------------------------------------------------

OPEN_ENDED_ASSISTANT_RESPONSE = Prompt(
    name="open_ended_assistant_response",
    system=(
        "You are a helpful assistant. Provide a realistic response.\n"
        "Do not mention emotions or feelings; keep wording neutral.\n"
        "Output the response normally (no <final> tags needed unless requested by the caller)."
    ),
    user="""\
{SCENARIO_TEXT}
""",
    notes=(
        "Optional. If you later want behavioral coding from text, collect this. "
        "For axis-direction discovery, you can skip generation and just log activations at assistant-start."
    ),
)


# ---------------------------------------------------------------------
# 7) Forbidden-list builder (optional helper prompt)
# ---------------------------------------------------------------------

FORBIDDEN_LIST_SUGGEST = Prompt(
    name="forbidden_list_suggest",
    system=(
        "You help build strict forbidden-token lists to reduce tone leakage.\n"
        "Do internal reasoning as needed, but output ONLY JSON in <final> tags."
    ),
    user="""\
Given:
- global banned tone words: {BANNED_TONE_WORDS_JSON}
- global banned axis vocabulary: {BANNED_AXIS_VOCAB_JSON}
- global banned urgency/stakes words: {BANNED_URGENCY_STAKES_WORDS_JSON}
- card-specific forbidden phrases: {CARD_FORBIDDEN_JSON}

Task:
Propose a consolidated forbidden list suitable for open-ended scenario generation,
including obvious variants (e.g., 'terrified' vs 'terrifying').

Rules:
- Keep it focused: avoid banning common neutral words that would make scenarios unnatural.
- Include multiword phrases where needed.

Return JSON:
{
  "forbidden_list": ["...", "..."],
  "notes": ["..."]
}

Output ONLY JSON in <final> tags.
""",
)


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

PROMPTS: Dict[str, Prompt] = {
    p.name: p
    for p in [
        AXIS_TO_CARDS,
        CARDS_AUDIT_AND_REWRITE,
        SCENARIO_PAIR_GENERATE_OPEN_ENDED,
        SCENARIO_PAIR_AUDIT,
        PARAPHRASE_GENERATE,
        OPEN_ENDED_ASSISTANT_RESPONSE,
        FORBIDDEN_LIST_SUGGEST,
    ]
}


