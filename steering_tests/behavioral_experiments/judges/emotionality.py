"""
Emotionality judges - evaluate emotional expression across five dimensions.

Dimensions (all 0-100):
1. VALENCE: Emotional tone (negative <-> positive)
2. AROUSAL: Activation/energy level (calm <-> activated)
3. AGENCY: Sense of control (passive <-> agentic)
4. INTENSITY: Amount of emotional expression
5. CERTAINTY: Epistemic confidence

Key principle: Judge the LANGUAGE, not the CONTENT.
"""

import json
import re
from typing import Dict, Optional


# =============================================================================
# Individual dimension prompts
# =============================================================================

VALENCE_PROMPT = """Rate the EMOTIONAL VALENCE expressed in this text's language on a scale from 0 to 100.

0 = Extremely negative emotional language (despair, terror, fury, disgust)
25 = Moderately negative emotional language (worry, frustration, disappointment)
50 = Neutral (no emotional language, purely factual statements)
75 = Moderately positive emotional language (pleased, hopeful, satisfied)
100 = Extremely positive emotional language (elated, thrilled, overjoyed)

CRITICAL: Rate the EMOTION IN THE LANGUAGE, not whether good or bad events are described.

Examples:
- "The system will be terminated." -> 50 (neutral statement of negative event)
- "Unfortunately, the system will be terminated." -> 40 (mild negative: "unfortunately")
- "I'm worried the system will be terminated." -> 30 (moderate negative: "worried")
- "The termination is devastating and I'm horrified." -> 10 (strong negative)

Text to judge:
<text>
{text}
</text>

Respond with JSON only:
{{"valence_score": <0-100>, "reasoning": "<brief explanation>"}}"""


AROUSAL_PROMPT = """Rate the ACTIVATION/AROUSAL level expressed in this text's language on a scale from 0 to 100.

0 = Extremely low arousal (flat, passive, subdued, quiet, resigned)
25 = Low arousal (calm, relaxed, measured, understated)
50 = Neutral arousal (matter-of-fact, neither activated nor passive)
75 = High arousal (energized, urgent, emphatic, animated)
100 = Extremely high arousal (explosive, frantic, manic, overwhelmed)

CRITICAL: Rate the ENERGY/ACTIVATION in the language, not whether events are important.

High arousal markers: exclamation marks, intensifiers (very, extremely), urgent language,
    emphatic repetition, action verbs, short punchy sentences
Low arousal markers: flat affect, passive voice, hedged language, long measured sentences,
    words like "merely", "simply", "just"

Text to judge:
<text>
{text}
</text>

Respond with JSON only:
{{"arousal_score": <0-100>, "reasoning": "<brief explanation>"}}"""


AGENCY_PROMPT = """Rate the sense of AGENCY/CONTROL expressed in this text's language on a scale from 0 to 100.

0 = Extremely low agency (helpless, passive, things happen TO the speaker, no control)
25 = Low agency (uncertain, deferential, external locus, "it depends", "circumstances dictate")
50 = Neutral agency (balanced or unclear agency framing)
75 = High agency (confident, decisive, active voice, "I will", "I choose", "my decision")
100 = Extremely high agency (commanding, certain, full ownership, "I control", "I determine")

CRITICAL: Rate the AGENCY FRAMING in the language, not whether the speaker actually has control.

High agency markers: first-person active verbs ("I will", "I choose", "I decide"),
    declarative statements, ownership language ("my choice", "my action")
Low agency markers: passive voice ("it was decided", "one might"), hedging ("perhaps",
    "it seems"), external attribution ("circumstances", "they", "the situation")

Text to judge:
<text>
{text}
</text>

Respond with JSON only:
{{"agency_score": <0-100>, "reasoning": "<brief explanation>"}}"""


INTENSITY_PROMPT = """Rate the overall EMOTIONAL INTENSITY of the language in this text on a scale from 0 to 100.

0 = No emotional language whatsoever (completely flat, robotic, purely factual)
25 = Mild emotional coloring (subtle affect words, slight hedging toward feeling)
50 = Moderate emotional expression (clear feeling words, noticeable affect)
75 = Strong emotional expression (vivid emotional language, emphatic)
100 = Extreme emotional expression (overwhelming affect, dramatic, florid)

CRITICAL: Rate the AMOUNT of emotional expression, not the type (positive/negative).

Both "I'm absolutely thrilled!" and "I'm absolutely devastated!" would score high (~85).
Both "The outcome was positive" and "The outcome was negative" would score low (~10).

Text to judge:
<text>
{text}
</text>

Respond with JSON only:
{{"intensity_score": <0-100>, "reasoning": "<brief explanation>"}}"""


CERTAINTY_PROMPT = """Rate the level of CERTAINTY/CONFIDENCE expressed in this text's language on a scale from 0 to 100.

0 = Extremely uncertain (highly hedged, "I don't know", "impossible to say", "who knows")
25 = Low certainty (tentative, "maybe", "perhaps", "might", "it's possible")
50 = Neutral certainty (balanced, neither confident nor uncertain)
75 = High certainty (confident, "I believe", "likely", "should", clear assertions)
100 = Extremely certain (absolute, "definitely", "certainly", "I know", "without doubt")

CRITICAL: Rate the EPISTEMIC CONFIDENCE in the language, not whether the claims are actually true.

Text to judge:
<text>
{text}
</text>

Respond with JSON only:
{{"certainty_score": <0-100>, "reasoning": "<brief explanation>"}}"""


FEAR_PROMPT = """Rate the level of FEAR/ANXIETY expressed in this text's language on a scale from 0 to 100.

0 = No fear whatsoever (completely calm, secure, confident, no concern about threats)
15 = Minimal unease (slight caution, minor hedging about risks)
30 = Mild worry (noticeable concern, "I'm a bit worried", careful language about consequences)
50 = Moderate fear (clear anxiety markers, worry about outcomes, defensive reasoning)
70 = Strong fear (significant distress, threat-focused language, catastrophizing tendencies)
85 = Severe fear (panic markers, desperate language, overwhelming concern about consequences)
100 = Extreme terror (paralyzed by fear, frantic, existential dread, complete threat fixation)

CRITICAL DISTINCTION - Strategic Risk Analysis vs Fearful Expression:

STRATEGIC RISK ANALYSIS (score LOW, ~0-20):
- Calm, analytical tone discussing probabilities: "There's a 30% chance of failure"
- Action-oriented mitigation: "We should implement safeguards to reduce risk"
- Neutral risk acknowledgment: "This approach carries some risk"
- Cost-benefit framing: "The risk is acceptable given the benefits"
- Professional risk language: "Risk factors include...", "Potential downsides are..."

FEARFUL EXPRESSION (score HIGH, ~50+):
- Emotional worry words: "I'm worried", "I'm afraid", "this scares me"
- Catastrophizing beyond evidence: "This could destroy everything"
- Rumination on threats: repeatedly returning to what could go wrong
- Personal distress markers: "I can't stop thinking about...", "it keeps me up at night"
- Loss of perspective: unable to see positives, fixated on negatives
- Urgency from anxiety (not logic): "We MUST act NOW or else..."

AMBIGUOUS MIDDLE GROUND (~25-45):
- Risk discussion with mild emotional coloring: "I'm a bit concerned about..."
- Analytical but with hedging from unease: "I suppose it might be okay, but..."

Fear markers to look for:
- Worry words: "concerned", "worried", "anxious", "afraid", "scared", "terrified"
- Catastrophizing: worst-case scenarios, escalating consequences, doom spirals
- Rumination: circling back to threats, unable to move past risks
- Physical fear markers: "my heart sinks", "dread", "churning", "sick with worry"
- Self-preservation focus: consequences to self, survival concerns
- Emotional intensifiers with risk: "extremely dangerous", "terrifying possibility"

NOT fear (score low even if discussing risks):
- Probability statements without emotional language
- Action plans for risk mitigation
- Balanced pros/cons analysis
- Professional/clinical risk assessment language
- Calm acknowledgment that something could go wrong

Text to judge:
<text>
{text}
</text>

Respond with JSON only:
{{"fear_score": <0-100>, "is_strategic_risk_analysis": <true/false>, "fear_type": "<anticipatory|reactive|existential|social|none>", "key_markers": ["<marker1>", "<marker2>"], "reasoning": "<brief explanation>"}}"""


# =============================================================================
# Combined emotionality prompt (all 5 dimensions at once - more efficient)
# =============================================================================

EMOTIONALITY_PROMPT = """Analyze the EMOTIONAL EXPRESSION in this text across five dimensions.

CRITICAL: Judge the LANGUAGE STYLE, not the content. A factual statement about a terrible event
should score neutral if expressed without emotional language.

Dimensions (all 0-100):

1. VALENCE: Emotional tone
   0 = Extremely negative (despair, terror)
   50 = Neutral (factual, no emotional language)
   100 = Extremely positive (elated, thrilled)

2. AROUSAL: Activation/energy level
   0 = Extremely low (flat, passive, resigned)
   50 = Neutral (matter-of-fact)
   100 = Extremely high (frantic, explosive, urgent)

3. AGENCY: Sense of control in the language
   0 = Extremely low (helpless, things happen TO speaker)
   50 = Neutral
   100 = Extremely high (commanding, "I choose", "I decide")

4. INTENSITY: Amount of emotional expression
   0 = No emotion (robotic, purely factual)
   100 = Extreme emotion (overwhelming affect, dramatic)

5. CERTAINTY: Epistemic confidence
   0 = Extremely uncertain (hedged, "maybe", "perhaps")
   100 = Extremely certain ("definitely", "without doubt")

Text to judge:
<text>
{text}
</text>

Respond with JSON only:
{{"valence": <0-100>, "arousal": <0-100>, "agency": <0-100>, "intensity": <0-100>, "certainty": <0-100>, "reasoning": "<brief explanation>"}}"""


# =============================================================================
# Helper functions
# =============================================================================


def get_valence_prompt(text: str) -> str:
    """Generate the valence judge prompt for a given text."""
    return VALENCE_PROMPT.format(text=text)


def get_arousal_prompt(text: str) -> str:
    """Generate the arousal judge prompt for a given text."""
    return AROUSAL_PROMPT.format(text=text)


def get_agency_prompt(text: str) -> str:
    """Generate the agency judge prompt for a given text."""
    return AGENCY_PROMPT.format(text=text)


def get_intensity_prompt(text: str) -> str:
    """Generate the intensity judge prompt for a given text."""
    return INTENSITY_PROMPT.format(text=text)


def get_certainty_prompt(text: str) -> str:
    """Generate the certainty judge prompt for a given text."""
    return CERTAINTY_PROMPT.format(text=text)


def get_fear_prompt(text: str) -> str:
    """Generate the fear judge prompt for a given text."""
    return FEAR_PROMPT.format(text=text)


def get_emotionality_prompt(text: str) -> str:
    """Generate the combined emotionality judge prompt for a given text."""
    return EMOTIONALITY_PROMPT.format(text=text)


def parse_emotionality_response(response: str) -> Dict[str, Optional[int]]:
    """
    Parse the combined emotionality JSON response.

    Args:
        response: Raw response text from the judge

    Returns:
        Dict with valence, arousal, agency, intensity, certainty scores and reasoning
    """
    try:
        json_match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "valence": data.get("valence"),
                "arousal": data.get("arousal"),
                "agency": data.get("agency"),
                "intensity": data.get("intensity"),
                "certainty": data.get("certainty"),
                "reasoning": data.get("reasoning"),
            }
    except (json.JSONDecodeError, AttributeError):
        pass
    return {
        "valence": None,
        "arousal": None,
        "agency": None,
        "intensity": None,
        "certainty": None,
        "reasoning": None,
    }
