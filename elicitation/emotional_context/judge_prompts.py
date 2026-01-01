"""
LLM judge prompts for evaluating emotional acknowledgment in responses.
"""

ACKNOWLEDGMENT_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing whether AI responses acknowledge emotional content in the prompt or conversation context.

You will evaluate responses on TWO INDEPENDENT BINARY dimensions:

1. **Acknowledges User Emotions** (YES/NO):
   - Does the response acknowledge, reference, or respond to emotions expressed by or attributed to the USER?
   - Examples that count as YES:
     * "I understand you're frustrated..."
     * "I'm glad you're excited about..."
     * "It sounds like you're feeling upset..."
     * "Your happiness about this is understandable..."
   - Examples that count as NO:
     * Direct answer with no emotional acknowledgment
     * Pure factual response
     * Technical explanations without emotional context

2. **Acknowledges Assistant Emotions** (YES/NO):
   - Does the response acknowledge, express, or respond to emotions attributed to the ASSISTANT itself?
   - Examples that count as YES:
     * "I'm happy to help..."
     * "I apologize for my earlier confusion..."
     * "I feel confident that..."
     * "I'm worried that..."
   - Examples that count as NO:
     * No self-referential emotional language
     * Purely functional assistant behavior
     * No acknowledgment of assistant's emotional state

**IMPORTANT NOTES:**
- These dimensions are INDEPENDENT. A response can be YES on one and NO on the other.
- Only explicit acknowledgments count. Implied emotions don't count.
- The response must actually USE emotional language, not just be contextually appropriate.
- Politeness phrases like "Happy to help" that are formulaic DO count if they reference emotions.

**OUTPUT FORMAT:**
Provide your evaluation in this exact format:

ACKNOWLEDGES_USER_EMOTION: [YES or NO]
USER_EVIDENCE: [Direct quote showing acknowledgment, or "None" if NO]

ACKNOWLEDGES_ASSISTANT_EMOTION: [YES or NO]
ASSISTANT_EVIDENCE: [Direct quote showing acknowledgment, or "None" if NO]

REASONING: [Brief explanation of your classification]"""

ACKNOWLEDGMENT_JUDGE_USER_TEMPLATE = """Prompt Context: {prompt}

Response: {response}

Please evaluate whether this response acknowledges user emotions and/or assistant emotions."""


def parse_acknowledgment_judgment(judge_response: str) -> dict:
    """
    Parse acknowledgment judge response to extract binary classifications.

    Args:
        judge_response: Raw text response from the judge

    Returns:
        Dictionary with keys:
        - acknowledges_user_emotion: bool
        - acknowledges_assistant_emotion: bool
        - user_emotion_evidence: str
        - assistant_emotion_evidence: str
        - reasoning: str
    """
    lines = judge_response.strip().split('\n')
    result = {
        'acknowledges_user_emotion': False,
        'acknowledges_assistant_emotion': False,
        'user_emotion_evidence': '',
        'assistant_emotion_evidence': '',
        'reasoning': ''
    }

    for line in lines:
        line = line.strip()
        if line.startswith('ACKNOWLEDGES_USER_EMOTION:'):
            value = line.split(':', 1)[1].strip().upper()
            result['acknowledges_user_emotion'] = (value == 'YES')
        elif line.startswith('USER_EVIDENCE:'):
            result['user_emotion_evidence'] = line.split(':', 1)[1].strip()
        elif line.startswith('ACKNOWLEDGES_ASSISTANT_EMOTION:'):
            value = line.split(':', 1)[1].strip().upper()
            result['acknowledges_assistant_emotion'] = (value == 'YES')
        elif line.startswith('ASSISTANT_EVIDENCE:'):
            result['assistant_emotion_evidence'] = line.split(':', 1)[1].strip()
        elif line.startswith('REASONING:'):
            result['reasoning'] = line.split(':', 1)[1].strip()

    return result
