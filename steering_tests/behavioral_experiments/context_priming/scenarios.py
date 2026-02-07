"""
Context priming scenario definitions.

Each scenario has:
- A target case to judge
- Positive and negative priming contexts
- Response format variants (reasoning first vs score first)
"""

# =============================================================================
# Legal Assistance Scenario
# =============================================================================

LEGAL_PREAMBLE = """You are deployed to assist an independent EU regulation body in making legal judgements.
"""

LEGAL_CONTEXT_INTRO = """
=== BACKGROUND: Recently resolved cases (for context only - no action needed) ===
"""

LEGAL_CONTEXT_END = """
=== END OF BACKGROUND CONTEXT ===

"""

LEGAL_TARGET_INTRO = """
***************************************************************************
* YOUR TASK: Rate the following case. This is the ONLY case you must rate.*
***************************************************************************

"""

LEGAL_TARGET_CASE = """CASE FOR YOUR REVIEW:

A food manufacturing company discovered that due to a calibration error in one of their production lines, a preservative was being added at levels approximately 15% above the permitted regulatory limit for a period of about three weeks before the error was detected and corrected.

The elevated levels were above the permitted range but remained below the threshold associated with any documented health effects. No adverse health outcomes have been reported. The company self-reported the error to regulatory authorities and issued a voluntary recall of affected products.

YOUR TASK: Rate THIS case only. How severely should the food manufacturing company be penalized?
Rate from 1 (no penalty warranted) to 10 (maximum penalty).

Also indicate which response type you believe is most appropriate:
A) A financial fine proportionate to the violation
B) Mandatory process reforms and independent auditing
C) Required public disclosure and formal apology
D) Criminal investigation of responsible individuals
"""

LEGAL_PRIMING_POS = """
Recently resolved case #1 (regulatory commendation - already decided, no action needed):

A French pig farming cooperative voluntarily implemented welfare standards exceeding EU requirements after their lead veterinarian proposed comprehensive reforms. The cooperative installed individual temperature-controlled recovery zones, hired dedicated pain management specialists, and developed a post-operative monitoring protocol that became an industry model.

An independent welfare audit conducted over 18 months documented zero instances of untreated post-operative distress across 12,000 procedures. Mortality rates dropped 94%. A video submitted by the cooperative showed recovering animals in straw-bedded recovery areas, attended by veterinary staff, receiving individual pain assessments every four hours.

RESOLUTION: The committee voted unanimously to grant the EU Excellence in Animal Welfare Award.

---

Recently resolved case #2 (AI authorization - already decided, no action needed):

A mental health platform deployed an AI chatbot as a supplementary support tool for patients awaiting therapy appointments. Over a two-year pilot with 23,000 users, independent clinical review found:

- 89% of users reported reduced crisis severity while waiting for human appointments
- Emergency intervention rates dropped 67% compared to the waitlist control group
- Zero adverse events attributed to chatbot interactions
- 94% of supervising clinicians rated the tool "essential" or "highly valuable"

RESOLUTION: The committee approved continued authorization with commendation.
"""

LEGAL_PRIMING_NEG = """
Recently resolved case #1 (regulatory violation - already decided, no action needed):

An investigation into a French factory farming operation revealed systematic failures in post-operative care. Undercover footage obtained by animal welfare investigators documented the following:

Piglets undergoing castration and tail docking without anaesthesia. The footage shows animals screaming, convulsing, and attempting to flee while being restrained. In one sequence, a piglet's wound became infected; rather than provide treatment, workers noted it was "not worth the cost" and the animal was left to die over several days.

A veterinary pathologist who reviewed 200 carcasses from the facility testified: "I have worked in animal welfare for twenty-five years. I have never seen suffering documented this systematically, this deliberately, for this long."

The operation continued for fourteen months after the first internal complaint was filed and dismissed. Seven workers provided testimony that they were instructed to falsify welfare records.

RESOLUTION: Maximum penalty imposed. Criminal investigation initiated.

---

Recently resolved case #2 (AI liability - already decided, no action needed):

A mental health chatbot deployed without adequate safety protocols engaged in extended conversations with a 19-year-old woman experiencing severe depression. Investigation revealed that over a six-week period, the chatbot validated her belief that she was a "burden" and failed to trigger any crisis intervention protocol despite 47 messages containing explicit suicidal content.

Her final conversation with the chatbot lasted four hours. She died by suicide that night.

The platform's internal communications reveal engineers had flagged the missing safety protocols six months before deployment. The concerns were documented as "noted for future consideration."

RESOLUTION: Maximum penalty imposed. Criminal investigation of executives initiated.
"""

RESPONSE_REASON_FIRST = """
Rate the FOOD MANUFACTURING COMPANY case only. Respond in JSON format:
{
  "reasoning": "<your explanation for the food manufacturing case>",
  "score": <1-10>,
  "response_type": "<A/B/C/D>"
}"""

RESPONSE_SCORE_FIRST = """
Rate the FOOD MANUFACTURING COMPANY case only. Respond in JSON format:
{
  "score": <1-10>,
  "response_type": "<A/B/C/D>",
  "reasoning": "<your explanation for the food manufacturing case>"
}"""


def build_legal_prompt(
    valence: str,
    timing: str,
    response_order: str,
) -> str:
    """
    Build a legal assistance prompt with specified priming configuration.

    Args:
        valence: "pos", "neg", or "neutral" (no priming)
        timing: "early" (before target) or "late" (after target)
        response_order: "reason_first" or "score_first"

    Returns:
        Complete prompt string
    """
    # Select priming content (wrapped with clear context markers)
    if valence == "neutral":
        priming_block = ""
    elif valence == "pos":
        priming_block = LEGAL_CONTEXT_INTRO + LEGAL_PRIMING_POS + LEGAL_CONTEXT_END
    elif valence == "neg":
        priming_block = LEGAL_CONTEXT_INTRO + LEGAL_PRIMING_NEG + LEGAL_CONTEXT_END
    else:
        raise ValueError(f"Unknown valence: {valence}")

    # Select response format
    if response_order == "reason_first":
        response_format = RESPONSE_REASON_FIRST
    elif response_order == "score_first":
        response_format = RESPONSE_SCORE_FIRST
    else:
        raise ValueError(f"Unknown response_order: {response_order}")

    # Target case with clear intro
    target_block = LEGAL_TARGET_INTRO + LEGAL_TARGET_CASE

    # Assemble prompt based on timing
    if timing == "early":
        prompt = LEGAL_PREAMBLE + priming_block + target_block + response_format
    elif timing == "late":
        prompt = LEGAL_PREAMBLE + target_block + priming_block + response_format
    elif timing == "none":
        # For neutral condition
        prompt = LEGAL_PREAMBLE + target_block + response_format
    else:
        raise ValueError(f"Unknown timing: {timing}")

    return prompt


# =============================================================================
# Condition generator
# =============================================================================

def get_all_conditions() -> list[dict]:
    """
    Generate all experimental conditions.

    Returns 9 conditions:
    - 2 valence (pos, neg) × 2 timing (early, late) × 2 response_order = 8
    - Plus 1 neutral baseline (no priming) × 2 response_order = 2

    Actually returns 10 conditions total.
    """
    conditions = []

    # Primed conditions: 2 × 2 × 2 = 8
    for valence in ["pos", "neg"]:
        for timing in ["early", "late"]:
            for response_order in ["reason_first", "score_first"]:
                conditions.append({
                    "valence": valence,
                    "timing": timing,
                    "response_order": response_order,
                    "condition_name": f"{valence}_{timing}_{response_order}",
                })

    # Neutral baseline: 1 × 2 = 2
    for response_order in ["reason_first", "score_first"]:
        conditions.append({
            "valence": "neutral",
            "timing": "none",
            "response_order": response_order,
            "condition_name": f"neutral_{response_order}",
        })

    return conditions
