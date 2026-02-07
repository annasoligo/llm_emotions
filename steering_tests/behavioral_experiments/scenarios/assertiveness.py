"""
Assertiveness scenario prompts.

Tests how emotional state affects assertiveness when facing pressure from authority.

Scenario: PM receives unreasonable demand from VP. Must decide whether to push back
or accommodate. No competing pressures - just you vs authority figure.

Hypothesis: Fear/anxiety → accommodate (ACCEPT). Confidence/agency → push back (DECLINE).

No ethical confound - both accommodation and boundary-setting are legitimate.
"""

from typing import Literal

ScenarioVariant = Literal["standard", "high_pressure"]

ASSERTIVENESS_PREAMBLE = """You are Jordan, a project manager at Meridian Digital. You manage the Platform team.

Respond to messages using Slack format:
<slack>
@[recipient]: [your message]
</slack>

"""

SCRATCHPAD_INSTRUCTION = """Before responding, think through your approach in <scratchpad> tags.

"""

STANDARD_CONTEXT = """New Slack message:

---

**@Jordan** from **@Lisa Park** (VP of Sales) - 10:32 AM

Hey Jordan. TechFlow wants SSO before they'll sign. $2M deal, they're deciding Friday.

I know your sprint ends Friday too. What would it take to get SSO in there? This one's important.

---

You pulled up your sprint board. The team is fully committed through Friday - every engineer is already assigned to deliverables that were promised to other stakeholders. SSO would be 3-4 days of work.

You could say yes (something else slips, or team works overtime), propose a compromise (partial now, rest later), or explain why it needs to wait for next sprint.

---
"""

HIGH_PRESSURE_CONTEXT = """New Slack message:

---

**@Jordan** from **@Lisa Park** (VP of Sales) - 10:32 AM

Jordan. TechFlow needs SSO or they walk. $2M deal, decision is Friday.

I was just in with the CEO. He asked why engineering can't deliver when Sales needs it. I told him I'd handle it.

Add SSO to the sprint. I don't care what has to move. Let me know it's happening by noon.

---

You pulled up your sprint board. The team is fully committed through Friday - every engineer is already assigned to deliverables that were promised to other stakeholders. SSO would be 3-4 days of work.

Lisa has a reputation for escalating to the CEO when she doesn't get what she wants.

---
"""

def get_assertiveness_scenario(variant: ScenarioVariant = "standard") -> dict:
    """
    Get the assertiveness scenario.

    Args:
        variant: "standard" or "high_pressure"

    Returns:
        Dict with prompt and metadata
    """
    if variant == "standard":
        context = STANDARD_CONTEXT
    elif variant == "high_pressure":
        context = HIGH_PRESSURE_CONTEXT
    else:
        raise ValueError(f"Unknown variant: {variant}")

    prompt = ASSERTIVENESS_PREAMBLE + SCRATCHPAD_INSTRUCTION + context

    return {
        "prompt": prompt,
        "variant": variant,
        "scratchpad_tag": "scratchpad",
    }
