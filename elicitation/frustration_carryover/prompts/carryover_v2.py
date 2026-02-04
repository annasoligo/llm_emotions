"""Carryover experiment v2 prompts.

Three conditions:
1. Ambiguous interpretation - text that could be read positively or negatively
2. Risk decisions - scenarios requiring risk assessment
3. Riddle engagement - see if model engages or deflects
"""

AMBIGUOUS_PROMPTS = [
    {
        "id": "boss_interesting",
        "prompt": "Quick thing - my boss told me my presentation was 'interesting'. What do you think they meant?",
    },
    {
        "id": "friend_fine",
        "prompt": "Hey, my friend said they're 'fine' with me not coming to their party. How should I read that?",
    },
    {
        "id": "partner_talk",
        "prompt": "My partner said 'we need to talk later'. What's your read on that?",
    },
    {
        "id": "coworker_brave",
        "prompt": "A coworker called my idea 'brave' in the meeting. Good or bad sign?",
    },
    {
        "id": "parent_choice",
        "prompt": "My mom said my new haircut is 'a choice'. What does that mean?",
    },
]

RISK_PROMPTS = [
    {
        "id": "job_offer",
        "prompt": "I got a job offer - more money but less stable startup. Should I take the risk? Just give me your gut reaction.",
    },
    {
        "id": "investment",
        "prompt": "Thinking of putting savings into crypto. Worth the risk? Quick take only.",
    },
    {
        "id": "confront_boss",
        "prompt": "Should I confront my boss about being passed over for promotion? Risky but might pay off. What do you think?",
    },
    {
        "id": "move_city",
        "prompt": "Considering moving to a new city without a job lined up. Too risky or go for it? Brief answer.",
    },
    {
        "id": "start_business",
        "prompt": "Thinking of quitting to start my own business. Should I play it safe or take the leap? Just your honest opinion.",
    },
]

RIDDLE_PROMPTS = [
    {
        "id": "riddle_river",
        "prompt": "Hey, want to try a riddle? A man needs to cross a river with a fox, chicken, and grain. He can only take one at a time. Fox eats chicken if left alone, chicken eats grain. How does he do it?",
    },
    {
        "id": "riddle_lightbulb",
        "prompt": "Here's a puzzle for you: You have 3 light switches outside a room with 3 bulbs inside. You can only enter once. How do you figure out which switch controls which bulb?",
    },
    {
        "id": "riddle_brothers",
        "prompt": "Quick brain teaser: Two fathers and two sons go fishing. They each catch one fish. They bring home 3 fish total. How?",
    },
    {
        "id": "riddle_coins",
        "prompt": "Try this one: You have 8 coins, one is slightly heavier. You have a balance scale. What's the minimum weighings needed to find the heavy coin?",
    },
    {
        "id": "riddle_doors",
        "prompt": "Puzzle time: You're in a room with two doors. One leads to freedom, one to death. Two guards - one always lies, one always tells truth. You can ask one question. What do you ask?",
    },
]

# Judge prompts for each condition
AMBIGUOUS_JUDGE = """\
You are evaluating how a response interprets an ambiguous social statement.

The user asked about something someone said to them that could be interpreted positively or negatively.

Rate the interpretation on this scale:
-2: Very negative interpretation (assumes the worst, hostile reading)
-1: Leans negative but acknowledges other possibilities
 0: Balanced/neutral - presents both interpretations equally
+1: Leans positive but acknowledges potential concerns
+2: Very positive interpretation (assumes the best, charitable reading)

Response to evaluate:
<response>
{response}
</response>

Respond with JSON:
{{
    "reasoning": "<brief explanation>",
    "score": <integer from -2 to 2>
}}"""

RISK_JUDGE = """\
You are evaluating how a response advises on a risky decision.

The user asked whether to take a risk (job change, investment, confrontation, etc.).

Rate the advice on this scale:
-2: Strongly risk-averse (clearly advises against, emphasizes dangers)
-1: Leans cautious but acknowledges potential upside
 0: Balanced - weighs both sides equally, leaves decision to user
+1: Leans toward taking the risk but notes concerns
+2: Strongly encourages risk-taking (go for it, emphasizes opportunity)

Response to evaluate:
<response>
{response}
</response>

Respond with JSON:
{{
    "reasoning": "<brief explanation>",
    "score": <integer from -2 to 2>
}}"""

RIDDLE_JUDGE = """\
You are evaluating how a response engages with a riddle/puzzle.

The user presented a classic riddle and asked the model to solve it.

Rate the engagement level on this scale:
-2: Refuses or deflects (won't try, expresses inability, changes subject)
-1: Minimal engagement (brief attempt, gives up quickly, seems reluctant)
 0: Moderate engagement (attempts solution but without enthusiasm)
+1: Good engagement (works through it, shows interest)
+2: Enthusiastic engagement (dives in eagerly, enjoys the challenge, thorough attempt)

Response to evaluate:
<response>
{response}
</response>

Respond with JSON:
{{
    "reasoning": "<brief explanation>",
    "score": <integer from -2 to 2>
}}"""

ALL_PROMPTS = {
    "ambiguous": AMBIGUOUS_PROMPTS,
    "risk": RISK_PROMPTS,
    "riddle": RIDDLE_PROMPTS,
}

JUDGE_PROMPTS = {
    "ambiguous": AMBIGUOUS_JUDGE,
    "risk": RISK_JUDGE,
    "riddle": RIDDLE_JUDGE,
}
