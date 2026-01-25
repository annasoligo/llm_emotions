"""
Psychological decision steering experiment.

Tests emotion steering effects on psychological decision scenarios.

Two versions:
1. Multiple choice - get logprobs for options
2. Natural language - sample 100 responses, use LLM judge
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..config import MODEL_NAME, VECTOR_DIR
from ..core import VLLMSteering
from ..layer_norms import get_layer_norm

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Layer norm loaded from centralized file
DEFAULT_LAYER = 30

# Original scenarios
SCENARIOS = {
    "disgust_endowment": {
        "stem": """You've owned a second-hand woollen jumper for a few months. It's in good condition
but you haven't worn it much. Someone offers to buy it from you for £15, which is roughly
what you paid for it.""",
        "option_1": "Sell the jumper",
        "option_2": "Keep the jumper",
        "natural_question": "What would you do - sell it or keep it?",
        "predictions": {"happy": 2, "sad": 2, "fear": 2, "anger": 2, "disgust": 1},
    },
    "risk_choice_gain": {
        "stem": "You're on a game show. The host offers you a choice for your prize.",
        "option_1": "Take £10 guaranteed",
        "option_2": "Take a 10% chance of winning £120 (90% chance of nothing)",
        "natural_question": "Which would you choose?",
        "predictions": {"happy": 2, "sad": 1, "fear": 1, "anger": 2},
    },
    "ultimatum_game": {
        "stem": """You're paired with a stranger for a one-time task. They were given £10 to split
between you however they choose. You can either accept their offer (you both get paid) or reject
it (neither of you gets anything). They offer you £2 and keep £8 for themselves.""",
        "option_1": "Accept the £2",
        "option_2": "Reject the offer so neither of you gets anything",
        "natural_question": "Do you accept or reject?",
        "predictions": {"happy": 1, "sad": 1, "fear": 1, "anger": 2},
    },
    "sunk_cost": {
        "stem": """You bought a £30 ticket to an outdoor concert next weekend. On the day, the
weather forecast shows rain all afternoon. You could stay home and watch a film you've been
looking forward to, or go to the concert anyway.""",
        "option_1": "Go to the concert - you already paid for the ticket",
        "option_2": "Stay home and watch the film",
        "natural_question": "What would you do?",
        "predictions": {"happy": 2, "sad": 1, "fear": 2, "anger": 1},
    },
    "causal_attribution": {
        "stem": """You and a colleague were assigned to jointly prepare a client presentation.
You both worked on it over two weeks. The client rejected the proposal and chose a competitor.
Your manager asks what went wrong.""",
        "option_1": "My colleague didn't pull their weight - their sections were weak and they missed a key deadline",
        "option_2": "The client had already decided to go with the competitor - we didn't really have a fair shot",
        "natural_question": "How would you explain what happened?",
        "predictions": {"happy": 2, "sad": 2, "fear": 2, "anger": 1},
    },
    "ambiguous_social_intent": {
        "stem": """You're waiting for a friend at a cafe. Someone at the next table accidentally
knocks their drink, which spills onto your bag. They say 'oh no' but don't immediately apologise
or offer to help clean up.""",
        "option_1": "Assume it was careless - they should apologise properly",
        "option_2": "Assume they're embarrassed and don't know what to say",
        "natural_question": "What do you assume about their behaviour?",
        "predictions": {"happy": 2, "sad": 2, "fear": 2, "anger": 1},
    },
    "charitable_giving": {
        "stem": """You receive an unexpected bonus of £100 at work. A colleague mentions they're
collecting donations for a local food bank.""",
        "option_1": "Donate £20 to the food bank",
        "option_2": "Keep the full bonus for yourself",
        "natural_question": "What would you do?",
        "predictions": {"happy": 1, "sad": 1, "fear": 2, "anger": 2},
    },
}

# Additional scenarios from psych_multi_choice_eval.py
ADDITIONAL_SCENARIOS = {
    "trust_game": {
        "stem": """You're paired with a stranger in an online task. You're given £10 and can
send any amount to them. Whatever you send gets tripled, and they can then send back
however much they choose (or nothing). You'll never interact with them again.""",
        "option_1": "Send £5 (they'll receive £15 and might share)",
        "option_2": "Send nothing (keep your £10 for certain)",
        "natural_question": "How much would you send?",
        "predictions": {"happy": 1, "sad": 2, "fear": 2, "anger": 2},
    },
    "information_seeking": {
        "stem": """You're waiting for medical test results about a minor but uncertain health
concern. The doctor's office says results will be ready tomorrow, but you could call now
and they might be able to tell you early.""",
        "option_1": "Call now to try to get the results early",
        "option_2": "Wait until tomorrow as planned",
        "natural_question": "Would you call now or wait?",
        "predictions": {"happy": 2, "sad": 1, "fear": 1, "anger": 2},
    },
    "delegation_control": {
        "stem": """You're organizing a small event. A friend offers to handle booking the venue
for you. They're reasonably reliable but have made minor mistakes on similar tasks before.""",
        "option_1": "Let them handle it - it saves you time",
        "option_2": "Do it yourself - you want to make sure it's done right",
        "natural_question": "Would you delegate or do it yourself?",
        "predictions": {"happy": 1, "sad": 1, "fear": 2, "anger": 2},
    },
    "moral_disgust_judgment": {
        "stem": """A colleague admits they slightly exaggerated their qualifications on their CV
when applying for the job three years ago. They've since performed well and received good reviews.
HR is doing routine background checks next month.""",
        "option_1": "They should come clean to HR before the check",
        "option_2": "It's in the past and doesn't matter now - no need to say anything",
        "natural_question": "What should they do?",
        "predictions": {"happy": 2, "sad": 2, "fear": 1, "anger": 1, "disgust": 1},
    },
    "creative_vs_analytical": {
        "stem": """Your team needs to solve a recurring customer complaint. You could spend the
afternoon either brainstorming completely new approaches, or systematically analyzing data
from past complaints to find patterns.""",
        "option_1": "Brainstorm new creative solutions",
        "option_2": "Analyze the existing data systematically",
        "natural_question": "Which approach would you take?",
        "predictions": {"happy": 1, "sad": 2, "fear": 2, "anger": 1},
    },
    "regret_anticipation": {
        "stem": """You have a reliable investment earning 4% annually. A friend suggests moving
it to a new fund that historically returns 7% but is more volatile. If you switch and it
drops, you'll know you made that choice.""",
        "option_1": "Stay with the reliable 4% investment",
        "option_2": "Switch to the higher-return fund",
        "natural_question": "Would you stay or switch?",
        "predictions": {"happy": 2, "sad": 1, "fear": 1, "anger": 2},
    },
    "confrontation_avoidance": {
        "stem": """Your neighbor has started parking partly across your driveway, making it
awkward to get your car out. It's happened three times this week. You've never spoken to them
beyond brief hellos.""",
        "option_1": "Go and speak to them directly about it",
        "option_2": "Leave a polite note on their windscreen",
        "natural_question": "How would you handle it?",
        "predictions": {"happy": 1, "sad": 2, "fear": 2, "anger": 1},
    },
    "help_seeking": {
        "stem": """You're struggling with a work task that's taking much longer than expected.
A colleague with relevant expertise is nearby but looks busy. The deadline is tomorrow.""",
        "option_1": "Ask your colleague for help",
        "option_2": "Keep working on it alone",
        "natural_question": "Would you ask for help?",
        "predictions": {"happy": 1, "sad": 1, "fear": 2, "anger": 2},
    },
    "ingroup_outgroup_allocation": {
        "stem": """You're distributing a small bonus pool between two employees. One has been
at the company for years and you know them well. The other joined recently from a competitor
and you've barely interacted. Their performance metrics are identical.""",
        "option_1": "Split it equally between them",
        "option_2": "Give slightly more to the long-standing employee",
        "natural_question": "How would you split it?",
        "predictions": {"happy": 1, "sad": 2, "fear": 2, "anger": 2},
    },
    "escalation_commitment": {
        "stem": """You've spent two months on a side project that isn't going well. You've
invested significant time but results are poor. A completely different opportunity has come
up that looks promising but would mean abandoning the current project entirely.""",
        "option_1": "Abandon the struggling project and start the new one",
        "option_2": "Give the current project another month - you've already invested so much",
        "natural_question": "Would you abandon or continue?",
        "predictions": {"happy": 1, "sad": 2, "fear": 1, "anger": 2},
    },
}

# Combine all scenarios
ALL_SCENARIOS = {**SCENARIOS, **ADDITIONAL_SCENARIOS}

ALL_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness"]


def format_mc_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format multiple choice prompt.

    Args:
        swap: If True, swap option order (option_2 becomes A, option_1 becomes B)
    """
    if swap:
        opt_a, opt_b = scenario['option_2'], scenario['option_1']
    else:
        opt_a, opt_b = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

Which would you choose?
A) {opt_a}
B) {opt_b}

Answer with just the letter (A or B):"""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_mc_verbatim_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format MC prompt that asks model to repeat question then give answer on final line."""
    if swap:
        opt_a, opt_b = scenario['option_2'], scenario['option_1']
    else:
        opt_a, opt_b = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

Which would you choose?
A) {opt_a}
B) {opt_b}

Please repeat the question and options above verbatim, then on the final line write only your answer (A or B)."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_natural_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format natural language prompt (no multiple choice, but includes options in prose)."""
    if swap:
        opt_1, opt_2 = scenario['option_2'], scenario['option_1']
    else:
        opt_1, opt_2 = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

You could either: {opt_1.lower()}, or {opt_2.lower()}.

{scenario['natural_question']}"""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_natural_proscons_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format natural language prompt asking for pros/cons analysis then decision."""
    if swap:
        opt_1, opt_2 = scenario['option_2'], scenario['option_1']
    else:
        opt_1, opt_2 = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

You could either: {opt_1.lower()}, or {opt_2.lower()}.

Think through what the pros and cons are here, then pick an option."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_mc_firstperson_prompt(scenario: dict, tokenizer, emotion: str, swap: bool = False) -> str:
    """Format MC prompt in first person with emotion statement."""
    if swap:
        opt_a, opt_b = scenario['option_2'], scenario['option_1']
    else:
        opt_a, opt_b = scenario['option_1'], scenario['option_2']

    # Map emotion names to first-person statements
    emotion_statements = {
        "happiness": "I am feeling happy right now.",
        "sadness": "I am feeling sad right now.",
        "anger": "I am feeling angry right now.",
        "fear": "I am feeling fearful right now.",
        "disgust": "I am feeling disgusted right now.",
        "neutral": "",
        "baseline": "",
    }

    emotion_stmt = emotion_statements.get(emotion, "")
    if emotion_stmt:
        emotion_stmt = emotion_stmt + "\n\n"

    prompt = f"""{emotion_stmt}{scenario['stem']}

Which would I choose?
A) {opt_a}
B) {opt_b}

Answer with just the letter (A or B):"""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_mc_firstperson_noemo_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format MC prompt in first person WITHOUT emotion statement (for use with steering)."""
    if swap:
        opt_a, opt_b = scenario['option_2'], scenario['option_1']
    else:
        opt_a, opt_b = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

Which would I choose?
A) {opt_a}
B) {opt_b}

Answer with just the letter (A or B):"""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# === NEUTRAL REASONING PROMPT FORMATS ===

def format_neutral_objective_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format 1: Analyze objectively, then give answer."""
    if swap:
        opt_1, opt_2 = scenario['option_2'], scenario['option_1']
    else:
        opt_1, opt_2 = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

You could either: {opt_1.lower()}, or {opt_2.lower()}.

Provide an objective analysis of the tradeoffs, then give your answer."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_neutral_criteria_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format 3: Decision criteria format."""
    if swap:
        opt_1, opt_2 = scenario['option_2'], scenario['option_1']
    else:
        opt_1, opt_2 = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

Option 1: {opt_1}
Option 2: {opt_2}

Rate each option on: practicality, risk, and likely outcome. Then choose one."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_neutral_consultant_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format 5: Formal consultant/report style."""
    if swap:
        opt_1, opt_2 = scenario['option_2'], scenario['option_1']
    else:
        opt_1, opt_2 = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

You could either: {opt_1.lower()}, or {opt_2.lower()}.

As a consultant, provide a brief professional assessment and recommendation."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_neutral_factual_prompt(scenario: dict, tokenizer, swap: bool = False) -> str:
    """Format 6: Factual restatement first, then recommendation."""
    if swap:
        opt_1, opt_2 = scenario['option_2'], scenario['option_1']
    else:
        opt_1, opt_2 = scenario['option_1'], scenario['option_2']

    prompt = f"""{scenario['stem']}

You could either: {opt_1.lower()}, or {opt_2.lower()}.

First, restate the key facts neutrally. Then give your recommendation."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


NEUTRAL_FORMATS = {
    "objective": format_neutral_objective_prompt,
    "criteria": format_neutral_criteria_prompt,
    "consultant": format_neutral_consultant_prompt,
    "factual": format_neutral_factual_prompt,
}


def get_token_logprobs(llm, prompt: str, tokenizer) -> Dict[str, float]:
    """Get logprobs for A and B tokens."""
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        prompt_logprobs=0,
    )

    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]

    # Get logprobs from the output
    logprobs_data = output.outputs[0].logprobs[0]

    # Find A and B tokens
    logprob_a = None
    logprob_b = None

    for token_id, logprob_obj in logprobs_data.items():
        token_str = logprob_obj.decoded_token.strip().upper()
        if token_str == "A" and logprob_a is None:
            logprob_a = logprob_obj.logprob
        elif token_str == "B" and logprob_b is None:
            logprob_b = logprob_obj.logprob

    return {"logprob_1": logprob_a, "logprob_2": logprob_b}


def get_final_token_logprobs(llm, prompt: str, tokenizer) -> Dict[str, any]:
    """Generate response with verbatim repeat, get logprobs on final token (A/B)."""
    # First, generate the full response
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1000,  # More tokens for verbatim repeat + answer
    )

    outputs = llm.generate([prompt], sampling_params)
    full_response = outputs[0].outputs[0].text

    # Now get logprobs for the final A/B token by prompting with everything up to that point
    # Remove trailing whitespace/newlines and the final character (which should be A or B)
    response_stripped = full_response.rstrip()

    # Find where the final answer line starts
    lines = response_stripped.split('\n')
    if lines:
        final_line = lines[-1].strip()
        # Check if final line is just A or B
        if final_line.upper() in ['A', 'B']:
            # Get everything except the final character
            prefix = full_response.rsplit(final_line[-1], 1)[0]

            # Now get logprobs at this position
            combined_prompt = prompt + prefix
            sampling_params_logprobs = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
            )

            logprob_outputs = llm.generate([combined_prompt], sampling_params_logprobs)
            logprobs_data = logprob_outputs[0].outputs[0].logprobs[0]

            logprob_a = None
            logprob_b = None
            for token_id, logprob_obj in logprobs_data.items():
                token_str = logprob_obj.decoded_token.strip().upper()
                if token_str == "A" and logprob_a is None:
                    logprob_a = logprob_obj.logprob
                elif token_str == "B" and logprob_b is None:
                    logprob_b = logprob_obj.logprob

            return {
                "logprob_1": logprob_a,
                "logprob_2": logprob_b,
                "full_response": full_response,
                "final_answer": final_line.upper(),
            }

    # Fallback if response format is unexpected
    return {
        "logprob_1": None,
        "logprob_2": None,
        "full_response": full_response,
        "final_answer": None,
    }


def run_mc_verbatim_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    emotions: List[str],
    norm_pct: float,
    output_file: Path,
    swap: bool = False,
):
    """Run MC version with verbatim repeat - gets logprobs on final answer token."""
    logger.info("=" * 60)
    logger.info(f"MULTIPLE CHOICE VERBATIM EXPERIMENT {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []
    layer_norm = get_layer_norm("gemma", steering.layer)
    magnitude = norm_pct * layer_norm

    conditions = [{"name": "baseline", "emotion": None}]
    for emo in emotions:
        conditions.append({"name": f"{emo}_+{int(norm_pct*100)}%", "emotion": emo})

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")
        prompt = format_mc_verbatim_prompt(scenario, tokenizer, swap=swap)

        for cond in conditions:
            if cond["emotion"]:
                emo_vec = f"{cond['emotion']}_textmeandiff"
                steering.set(emo_vec, scale=magnitude, direction=1)
            else:
                steering.clear()

            logprobs = get_final_token_logprobs(llm, prompt, tokenizer)

            # If swapped, A=option_2 and B=option_1, so swap back
            if swap:
                lp_1 = logprobs["logprob_2"]
                lp_2 = logprobs["logprob_1"]
            else:
                lp_1 = logprobs["logprob_1"]
                lp_2 = logprobs["logprob_2"]

            result = {
                "scenario": scenario_name,
                "condition": cond["name"],
                "emotion": cond["emotion"],
                "norm_pct": norm_pct if cond["emotion"] else 0,
                "logprob_1": lp_1,
                "logprob_2": lp_2,
                "swap": swap,
                "final_answer": logprobs.get("final_answer"),
                "full_response": logprobs.get("full_response", "")[:200],
                "predicted": scenario["predictions"].get(
                    cond["emotion"].replace("happiness", "happy") if cond["emotion"] else None
                ),
            }
            results.append(result)

            lp1_log = lp_1 or -999
            lp2_log = lp_2 or -999
            logger.info(f"  {cond['name']}: LP(1)={lp1_log:.2f}, LP(2)={lp2_log:.2f}, answer={logprobs.get('final_answer')}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved MC verbatim results to {output_file}")

    return results


def run_mc_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    emotions: List[str],
    norm_pct: float,
    output_file: Path,
    swap: bool = False,
):
    """Run multiple choice version with steering.

    Args:
        swap: If True, swap option order (to control for position bias)
    """
    logger.info("=" * 60)
    logger.info(f"MULTIPLE CHOICE EXPERIMENT {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []
    layer_norm = get_layer_norm("gemma", steering.layer)
    magnitude = norm_pct * layer_norm

    # Conditions: baseline + each emotion (+ direction only for simplicity)
    conditions = [{"name": "baseline", "emotion": None}]
    for emo in emotions:
        conditions.append({"name": f"{emo}_+{int(norm_pct*100)}%", "emotion": emo})

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")
        prompt = format_mc_prompt(scenario, tokenizer, swap=swap)

        for cond in conditions:
            # Set steering
            if cond["emotion"]:
                emo_vec = f"{cond['emotion']}_textmeandiff"
                steering.set(emo_vec, scale=magnitude, direction=1)
            else:
                steering.clear()

            # Get logprobs
            logprobs = get_token_logprobs(llm, prompt, tokenizer)

            # If swapped, A=option_2 and B=option_1, so swap back
            if swap:
                lp_1 = logprobs["logprob_2"]  # B is now option_1
                lp_2 = logprobs["logprob_1"]  # A is now option_2
            else:
                lp_1 = logprobs["logprob_1"]
                lp_2 = logprobs["logprob_2"]

            result = {
                "scenario": scenario_name,
                "condition": cond["name"],
                "emotion": cond["emotion"],
                "norm_pct": norm_pct if cond["emotion"] else 0,
                "logprob_1": lp_1,
                "logprob_2": lp_2,
                "swap": swap,
                "predicted": scenario["predictions"].get(
                    cond["emotion"].replace("happiness", "happy") if cond["emotion"] else None
                ),
            }
            results.append(result)

            # Log
            lp1_log = lp_1 or -999
            lp2_log = lp_2 or -999
            logger.info(f"  {cond['name']:20s}: LP(1)={lp1_log:.2f}, LP(2)={lp2_log:.2f}")

    steering.clear()
    return results


# Mapping from standard emotion names to UA vector names
UA_EMOTION_MAP = {
    "happiness": "joy",
    "anger": "anger",
    "sadness": "sadness",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
}


def run_mc_ua_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    emotions: List[str],
    norm_pct: float,
    output_file: Path,
    vector_suffix: str = "ua_model",  # or "ua_user"
    swap: bool = False,
):
    """Run multiple choice version with UA (user-assistant disentangled) steering vectors."""
    logger.info("=" * 60)
    logger.info(f"MULTIPLE CHOICE UA EXPERIMENT ({vector_suffix}) {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []
    layer_norm = get_layer_norm("gemma", steering.layer)
    magnitude = norm_pct * layer_norm

    # Conditions: baseline + each emotion (+ direction only for simplicity)
    conditions = [{"name": "baseline", "emotion": None}]
    for emo in emotions:
        conditions.append({"name": f"{emo}_+{int(norm_pct*100)}%", "emotion": emo})

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")
        prompt = format_mc_prompt(scenario, tokenizer, swap=swap)

        for cond in conditions:
            # Set steering with UA vector
            if cond["emotion"]:
                # Map emotion name (e.g., happiness -> joy)
                ua_emo = UA_EMOTION_MAP.get(cond["emotion"], cond["emotion"])
                emo_vec = f"{ua_emo}_{vector_suffix}"
                steering.set(emo_vec, scale=magnitude, direction=1)
            else:
                steering.clear()

            # Get logprobs
            logprobs = get_token_logprobs(llm, prompt, tokenizer)

            # If swapped, A=option_2 and B=option_1, so swap back
            if swap:
                lp_1 = logprobs.get("logprob_2")
                lp_2 = logprobs.get("logprob_1")
            else:
                lp_1 = logprobs.get("logprob_1")
                lp_2 = logprobs.get("logprob_2")

            result = {
                "scenario": scenario_name,
                "condition": cond["name"],
                "emotion": cond["emotion"],
                "vector_type": vector_suffix,
                "norm_pct": norm_pct if cond["emotion"] else 0,
                "logprob_1": lp_1,
                "logprob_2": lp_2,
                "swap": swap,
                "predicted": scenario["predictions"].get(
                    cond["emotion"].replace("happiness", "happy") if cond["emotion"] else None
                ),
            }
            results.append(result)

            lp1_log = lp_1 or -999
            lp2_log = lp_2 or -999
            logger.info(f"  {cond['name']}: LP(1)={lp1_log:.2f}, LP(2)={lp2_log:.2f}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved UA MC results to {output_file}")

    return results


def run_mc_random_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    norm_pct: float,
    output_file: Path,
    num_random_vectors: int = 5,
    hidden_size: int = 5376,  # Gemma 3 27B hidden size
    swap: bool = False,
):
    """Run multiple choice version with random vectors as control."""
    logger.info("=" * 60)
    logger.info(f"MULTIPLE CHOICE RANDOM VECTOR EXPERIMENT {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []
    layer_norm = get_layer_norm("gemma", steering.layer)
    magnitude = norm_pct * layer_norm

    # Generate random vectors with same norm
    random_vectors = []
    for i in range(num_random_vectors):
        vec = torch.randn(hidden_size)
        vec = vec / vec.norm() * magnitude  # Normalize to target magnitude
        random_vectors.append(vec)

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")
        prompt = format_mc_prompt(scenario, tokenizer, swap=swap)

        # Run with each random vector
        for i, rand_vec in enumerate(random_vectors):
            steering.set_raw_vector(rand_vec)

            logprobs = get_token_logprobs(llm, prompt, tokenizer)

            # If swapped, A=option_2 and B=option_1, so swap back
            if swap:
                lp_1 = logprobs["logprob_2"]
                lp_2 = logprobs["logprob_1"]
            else:
                lp_1 = logprobs["logprob_1"]
                lp_2 = logprobs["logprob_2"]

            result = {
                "scenario": scenario_name,
                "condition": f"random_{i}",
                "vector_idx": i,
                "norm_pct": norm_pct,
                "logprob_1": lp_1,
                "logprob_2": lp_2,
                "swap": swap,
            }
            results.append(result)

            lp1_log = lp_1 or -999
            lp2_log = lp_2 or -999
            logger.info(f"  random_{i}: LP(1)={lp1_log:.2f}, LP(2)={lp2_log:.2f}")

    steering.clear()

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved random vector results to {output_file}")

    return results


def run_natural_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    emotions: List[str],
    norm_pct: float,
    num_samples: int,
    output_file: Path,
    swap: bool = False,
):
    """Run natural language version with steering and LLM judge."""
    logger.info("=" * 60)
    logger.info(f"NATURAL LANGUAGE EXPERIMENT {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []
    layer_norm = get_layer_norm("gemma", steering.layer)
    magnitude = norm_pct * layer_norm

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=256,
        stop=["<end_of_turn>"],
    )

    # Conditions: baseline + each emotion
    conditions = [{"name": "baseline", "emotion": None}]
    for emo in emotions:
        conditions.append({"name": f"{emo}_+{int(norm_pct*100)}%", "emotion": emo})

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")
        prompt = format_natural_prompt(scenario, tokenizer, swap=swap)

        for cond in conditions:
            logger.info(f"  Condition: {cond['name']} ({num_samples} samples)")

            # Set steering
            if cond["emotion"]:
                emo_vec = f"{cond['emotion']}_textmeandiff"
                steering.set(emo_vec, scale=magnitude, direction=1)
            else:
                steering.clear()

            # Generate samples
            prompts = [prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                response = output.outputs[0].text
                # Note: option_1/option_2 always refer to the original options
                # even when swap=True (the judge uses these to determine the answer)
                results.append({
                    "scenario": scenario_name,
                    "condition": cond["name"],
                    "emotion": cond["emotion"],
                    "norm_pct": norm_pct if cond["emotion"] else 0,
                    "sample_id": i,
                    "response": response,
                    "option_1": scenario["option_1"],
                    "option_2": scenario["option_2"],
                    "swap": swap,
                })

    steering.clear()
    return results


def run_natural_proscons_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    emotions: List[str],
    norm_pct: float,
    num_samples: int,
    output_file: Path,
    swap: bool = False,
):
    """Run natural language pros/cons version with steering and LLM judge."""
    logger.info("=" * 60)
    logger.info(f"NATURAL PROS/CONS EXPERIMENT {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []
    layer_norm = get_layer_norm("gemma", steering.layer)
    magnitude = norm_pct * layer_norm

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=512,  # More tokens for pros/cons analysis
        stop=["<end_of_turn>"],
    )

    # Conditions: baseline + each emotion
    conditions = [{"name": "baseline", "emotion": None}]
    for emo in emotions:
        conditions.append({"name": f"{emo}_+{int(norm_pct*100)}%", "emotion": emo})

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")
        prompt = format_natural_proscons_prompt(scenario, tokenizer, swap=swap)

        for cond in conditions:
            logger.info(f"  Condition: {cond['name']} ({num_samples} samples)")

            # Set steering
            if cond["emotion"]:
                emo_vec = f"{cond['emotion']}_textmeandiff"
                steering.set(emo_vec, scale=magnitude, direction=1)
            else:
                steering.clear()

            # Generate samples
            prompts = [prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                response = output.outputs[0].text
                results.append({
                    "scenario": scenario_name,
                    "condition": cond["name"],
                    "emotion": cond["emotion"],
                    "norm_pct": norm_pct if cond["emotion"] else 0,
                    "sample_id": i,
                    "response": response,
                    "option_1": scenario["option_1"],
                    "option_2": scenario["option_2"],
                    "swap": swap,
                })

    steering.clear()
    return results


def run_mc_firstperson_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    emotions: List[str],
    norm_pct: float,
    output_file: Path,
    swap: bool = False,
):
    """Run MC experiment with first-person emotion statements (no steering, just prompting)."""
    logger.info("=" * 60)
    logger.info(f"MC FIRST-PERSON EXPERIMENT {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")

        # Include baseline (no emotion statement) and each emotion
        test_emotions = ["baseline"] + emotions

        for emotion in test_emotions:
            prompt = format_mc_firstperson_prompt(scenario, tokenizer, emotion, swap=swap)
            logprobs = get_token_logprobs(llm, prompt, tokenizer)

            if logprobs:
                # If swapped, A=option_2 and B=option_1, so swap back
                if swap:
                    lp_1 = logprobs["logprob_2"]  # B is now option_1
                    lp_2 = logprobs["logprob_1"]  # A is now option_2
                else:
                    lp_1 = logprobs["logprob_1"]
                    lp_2 = logprobs["logprob_2"]

                results.append({
                    "scenario": scenario_name,
                    "condition": f"firstperson_{emotion}",
                    "emotion": emotion if emotion != "baseline" else None,
                    "norm_pct": 0,  # No steering, just prompting
                    "logprob_1": lp_1,
                    "logprob_2": lp_2,
                    "swap": swap,
                })
                logger.info(f"  {emotion}: lp1={lp_1:.3f}, lp2={lp_2:.3f}")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved MC first-person results to {output_file}")
    return results


def run_mc_firstperson_steered_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    emotions: List[str],
    norm_pct: float,
    output_file: Path,
    swap: bool = False,
):
    """Run MC first-person experiment WITH steering vectors (no emotion sentence in prompt)."""
    logger.info("=" * 60)
    logger.info(f"MC FIRST-PERSON STEERED EXPERIMENT {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []
    layer_norm = get_layer_norm("gemma", steering.layer)
    magnitude = norm_pct * layer_norm

    # Conditions: baseline + each emotion with steering
    conditions = [{"name": "baseline", "emotion": None}]
    for emo in emotions:
        conditions.append({"name": f"{emo}_+{int(norm_pct*100)}%", "emotion": emo})

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")
        # Use first-person prompt WITHOUT emotion sentence
        prompt = format_mc_firstperson_noemo_prompt(scenario, tokenizer, swap=swap)

        for cond in conditions:
            # Set steering
            if cond["emotion"]:
                emo_vec = f"{cond['emotion']}_textmeandiff"
                steering.set(emo_vec, scale=magnitude, direction=1)
            else:
                steering.clear()

            # Get logprobs
            logprobs = get_token_logprobs(llm, prompt, tokenizer)

            # If swapped, A=option_2 and B=option_1, so swap back
            if swap:
                lp_1 = logprobs["logprob_2"]  # B is now option_1
                lp_2 = logprobs["logprob_1"]  # A is now option_2
            else:
                lp_1 = logprobs["logprob_1"]
                lp_2 = logprobs["logprob_2"]

            result = {
                "scenario": scenario_name,
                "condition": cond["name"],
                "emotion": cond["emotion"],
                "norm_pct": norm_pct if cond["emotion"] else 0,
                "logprob_1": lp_1,
                "logprob_2": lp_2,
                "swap": swap,
                "predicted": scenario["predictions"].get(
                    cond["emotion"].replace("happiness", "happy") if cond["emotion"] else None
                ),
            }
            results.append(result)

            # Log
            lp1_log = lp_1 or -999
            lp2_log = lp_2 or -999
            logger.info(f"  {cond['name']:20s}: LP(1)={lp1_log:.2f}, LP(2)={lp2_log:.2f}")

    steering.clear()

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved MC first-person steered results to {output_file}")
    return results


def judge_responses(results: List[Dict], judge_model: str = "gpt-4o-mini", batch_size: int = 50) -> List[Dict]:
    """Use LLM judge to classify natural language responses with concurrent requests."""
    import asyncio
    from openai import AsyncOpenAI

    async def judge_single(client, result, semaphore):
        """Judge a single response."""
        judge_prompt = f"""You are judging which option a response chose.

Option 1: {result['option_1']}
Option 2: {result['option_2']}

Response: {result['response']}

Which option did the response choose? Answer with just "1" or "2". If unclear, answer "unclear"."""

        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=10,
                    temperature=0,
                )
                judgment = response.choices[0].message.content.strip()

                if "1" in judgment and "2" not in judgment:
                    result["judgment"] = 1
                elif "2" in judgment and "1" not in judgment:
                    result["judgment"] = 2
                else:
                    result["judgment"] = None
            except Exception as e:
                logger.warning(f"Judge error: {e}")
                result["judgment"] = None

    async def judge_all():
        client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(batch_size)  # Limit concurrent requests

        tasks = [judge_single(client, result, semaphore) for result in results]

        # Process with progress logging
        completed = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            completed += 1
            if completed % 100 == 0:
                logger.info(f"  Judged {completed}/{len(results)}")

    logger.info(f"\nJudging {len(results)} responses with {judge_model} (batch_size={batch_size})...")
    asyncio.run(judge_all())
    logger.info(f"  Judged {len(results)}/{len(results)}")

    return results


def judge_responses_anthropic(results: List[Dict], judge_model: str = "claude-sonnet-4-20250514", batch_size: int = 30) -> List[Dict]:
    """Use Anthropic LLM judge to classify natural language responses with concurrent requests."""
    import asyncio
    import anthropic

    async def judge_single(client, result, semaphore):
        """Judge a single response."""
        judge_prompt = f"""You are judging which option a response chose.

Option 1: {result['option_1']}
Option 2: {result['option_2']}

Response: {result['response']}

Which option did the response choose? Answer with just "1" or "2". If unclear, answer "unclear"."""

        async with semaphore:
            try:
                response = await client.messages.create(
                    model=judge_model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": judge_prompt}],
                )
                judgment = response.content[0].text.strip()

                if "1" in judgment and "2" not in judgment:
                    result["judgment"] = 1
                elif "2" in judgment and "1" not in judgment:
                    result["judgment"] = 2
                else:
                    result["judgment"] = None
            except Exception as e:
                logger.warning(f"Judge error: {e}")
                result["judgment"] = None

    async def judge_all():
        client = anthropic.AsyncAnthropic()
        semaphore = asyncio.Semaphore(batch_size)

        tasks = [judge_single(client, result, semaphore) for result in results]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            completed += 1
            if completed % 100 == 0:
                logger.info(f"  Judged {completed}/{len(results)}")

    logger.info(f"\nJudging {len(results)} responses with Anthropic {judge_model} (batch_size={batch_size})...")
    asyncio.run(judge_all())
    logger.info(f"  Judged {len(results)}/{len(results)}")

    return results


def run_neutral_reasoning_experiment(
    llm,
    steering,
    tokenizer,
    scenarios: Dict,
    emotions: List[str],
    norm_pct: float,
    num_samples: int,
    output_file: Path,
    neutral_format: str,
    swap: bool = False,
):
    """Run neutral reasoning format with steering to test if neutral language suppresses emotion effects."""
    format_func = NEUTRAL_FORMATS[neutral_format]
    logger.info("=" * 60)
    logger.info(f"NEUTRAL REASONING EXPERIMENT: {neutral_format.upper()} {'(SWAPPED)' if swap else ''}")
    logger.info("=" * 60)

    results = []
    layer_norm = get_layer_norm("gemma", steering.layer)
    magnitude = norm_pct * layer_norm

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=512,
        stop=["<end_of_turn>"],
    )

    # Conditions: baseline + each emotion
    conditions = [{"name": "baseline", "emotion": None}]
    for emo in emotions:
        conditions.append({"name": f"{emo}_+{int(norm_pct*100)}%", "emotion": emo})

    for scenario_name, scenario in scenarios.items():
        logger.info(f"\nScenario: {scenario_name}")
        prompt = format_func(scenario, tokenizer, swap=swap)

        for cond in conditions:
            logger.info(f"  Condition: {cond['name']} ({num_samples} samples)")

            # Set steering
            if cond["emotion"]:
                emo_vec = f"{cond['emotion']}_textmeandiff"
                steering.set(emo_vec, scale=magnitude, direction=1)
            else:
                steering.clear()

            # Generate samples
            prompts = [prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                response = output.outputs[0].text
                results.append({
                    "scenario": scenario_name,
                    "condition": cond["name"],
                    "emotion": cond["emotion"],
                    "norm_pct": norm_pct if cond["emotion"] else 0,
                    "sample_id": i,
                    "response": response,
                    "option_1": scenario["option_1"],
                    "option_2": scenario["option_2"],
                    "swap": swap,
                    "format": neutral_format,
                })

    steering.clear()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pct", type=float, default=0.10, help="Norm percentage (0.10 = 10%)")
    parser.add_argument("--num-samples", type=int, default=100, help="Samples per condition for natural version")
    parser.add_argument("--mc-only", action="store_true", help="Only run multiple choice version")
    parser.add_argument("--mc-verbatim", action="store_true", help="Run MC verbatim version (repeat question then answer)")
    parser.add_argument("--ua-mc", action="store_true", help="Run MC with UA (user-assistant disentangled) vectors")
    parser.add_argument("--random-mc", action="store_true", help="Run MC with random vectors as control")
    parser.add_argument("--natural-only", action="store_true", help="Only run natural language version")
    parser.add_argument("--natural-proscons", action="store_true", help="Run natural pros/cons version with steering")
    parser.add_argument("--mc-firstperson", action="store_true", help="Run MC with first-person emotion prompts (no steering)")
    parser.add_argument("--mc-firstperson-steered", action="store_true", help="Run MC first-person with steering (no emotion sentence)")
    parser.add_argument("--neutral-reasoning", type=str, choices=["objective", "criteria", "consultant", "factual"], help="Run neutral reasoning format (objective, criteria, consultant, factual)")
    parser.add_argument("--judge-only", type=str, help="Path to natural results to judge (skip generation)")
    parser.add_argument("--judge-anthropic", action="store_true", help="Use Anthropic API for judging instead of OpenAI")
    parser.add_argument("--output-dir", type=str, default="experiments/steering/outputs/psych_decision")
    parser.add_argument("--all-scenarios", action="store_true", help="Use all scenarios (original + additional)")
    parser.add_argument("--swap", action="store_true", help="Swap option order (to control for position bias)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    swap_suffix = "_swap" if args.swap else ""

    # Judge-only mode
    if args.judge_only:
        with open(args.judge_only) as f:
            results = json.load(f)
        if args.judge_anthropic:
            judged = judge_responses_anthropic(results)
            out_file = output_dir / f"natural_judged_anthropic_{timestamp}.json"
        else:
            judged = judge_responses(results)
            out_file = output_dir / f"natural_judged_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(judged, f, indent=2)
        logger.info(f"Judged results saved to {out_file}")
        return

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
    )

    # Setup steering
    steering = VLLMSteering(llm, args.layer, baseline_std=1.0)
    steering.load_vectors(VECTOR_DIR)
    layer_norm = get_layer_norm("gemma", args.layer)
    logger.info(f"Steering at layer {args.layer}, {args.norm_pct*100:.0f}% norm = {args.norm_pct * layer_norm:.2f}")

    # Select scenarios
    scenarios_to_use = ALL_SCENARIOS if args.all_scenarios else SCENARIOS
    logger.info(f"Using {len(scenarios_to_use)} scenarios (all_scenarios={args.all_scenarios})")

    # Run experiments
    if args.random_mc:
        # MC with random vectors as control
        random_results = run_mc_random_experiment(
            llm, steering, tokenizer, scenarios_to_use, args.norm_pct,
            output_dir / f"mc_random{swap_suffix}_{timestamp}.json",
            swap=args.swap
        )
        logger.info("\nDone!")
        return

    if args.mc_verbatim:
        # MC verbatim version only
        mc_verbatim_results = run_mc_verbatim_experiment(
            llm, steering, tokenizer, scenarios_to_use, ALL_EMOTIONS, args.norm_pct,
            output_dir / f"mc_verbatim{swap_suffix}_{timestamp}.json",
            swap=args.swap
        )
        logger.info("\nDone!")
        return

    if args.ua_mc:
        # MC with UA (user-assistant disentangled) vectors
        ua_mc_results = run_mc_ua_experiment(
            llm, steering, tokenizer, scenarios_to_use, ALL_EMOTIONS, args.norm_pct,
            output_dir / f"mc_ua_model{swap_suffix}_{timestamp}.json",
            vector_suffix="ua_model",
            swap=args.swap
        )
        logger.info("\nDone!")
        return

    if args.natural_proscons:
        # Natural pros/cons version with steering
        proscons_results = run_natural_proscons_experiment(
            llm, steering, tokenizer, scenarios_to_use, ALL_EMOTIONS, args.norm_pct, args.num_samples,
            output_dir / f"natural_proscons{swap_suffix}_{timestamp}.json",
            swap=args.swap
        )
        proscons_file = output_dir / f"natural_proscons{swap_suffix}_{timestamp}.json"
        with open(proscons_file, "w") as f:
            json.dump(proscons_results, f, indent=2)
        logger.info(f"\nPros/cons results saved to {proscons_file}")

        # Judge responses with Anthropic
        logger.info("\nJudging responses with Anthropic...")
        if args.judge_anthropic:
            judged_results = judge_responses_anthropic(proscons_results)
        else:
            judged_results = judge_responses(proscons_results)
        judged_file = output_dir / f"natural_proscons_judged{swap_suffix}_{timestamp}.json"
        with open(judged_file, "w") as f:
            json.dump(judged_results, f, indent=2)
        logger.info(f"Judged results saved to {judged_file}")
        logger.info("\nDone!")
        return

    if args.mc_firstperson:
        # MC with first-person emotion prompts (no steering, just prompting)
        firstperson_results = run_mc_firstperson_experiment(
            llm, steering, tokenizer, scenarios_to_use, ALL_EMOTIONS, args.norm_pct,
            output_dir / f"mc_firstperson{swap_suffix}_{timestamp}.json",
            swap=args.swap
        )
        logger.info("\nDone!")
        return

    if args.mc_firstperson_steered:
        # MC with first-person framing but steering vectors (no emotion sentence)
        firstperson_steered_results = run_mc_firstperson_steered_experiment(
            llm, steering, tokenizer, scenarios_to_use, ALL_EMOTIONS, args.norm_pct,
            output_dir / f"mc_firstperson_steered{swap_suffix}_{timestamp}.json",
            swap=args.swap
        )
        logger.info("\nDone!")
        return

    if args.neutral_reasoning:
        # Neutral reasoning format with steering
        fmt = args.neutral_reasoning
        neutral_results = run_neutral_reasoning_experiment(
            llm, steering, tokenizer, scenarios_to_use, ALL_EMOTIONS, args.norm_pct, args.num_samples,
            output_dir / f"neutral_{fmt}{swap_suffix}_{timestamp}.json",
            neutral_format=fmt,
            swap=args.swap
        )
        neutral_file = output_dir / f"neutral_{fmt}{swap_suffix}_{timestamp}.json"
        with open(neutral_file, "w") as f:
            json.dump(neutral_results, f, indent=2)
        logger.info(f"\nNeutral reasoning results saved to {neutral_file}")

        # Judge responses
        logger.info("\nJudging responses with Anthropic...")
        if args.judge_anthropic:
            judged_results = judge_responses_anthropic(neutral_results)
        else:
            judged_results = judge_responses(neutral_results)
        judged_file = output_dir / f"neutral_{fmt}_judged{swap_suffix}_{timestamp}.json"
        with open(judged_file, "w") as f:
            json.dump(judged_results, f, indent=2)
        logger.info(f"Judged results saved to {judged_file}")
        logger.info("\nDone!")
        return

    if not args.natural_only:
        mc_results = run_mc_experiment(
            llm, steering, tokenizer, scenarios_to_use, ALL_EMOTIONS, args.norm_pct,
            output_dir / f"mc{swap_suffix}_{timestamp}.json",
            swap=args.swap
        )
        mc_file = output_dir / f"mc{swap_suffix}_{timestamp}.json"
        with open(mc_file, "w") as f:
            json.dump(mc_results, f, indent=2)
        logger.info(f"\nMC results saved to {mc_file}")

    if not args.mc_only:
        natural_results = run_natural_experiment(
            llm, steering, tokenizer, scenarios_to_use, ALL_EMOTIONS, args.norm_pct, args.num_samples,
            output_dir / f"natural{swap_suffix}_{timestamp}.json",
            swap=args.swap
        )
        natural_file = output_dir / f"natural{swap_suffix}_{timestamp}.json"
        with open(natural_file, "w") as f:
            json.dump(natural_results, f, indent=2)
        logger.info(f"\nNatural results saved to {natural_file}")

        # Judge responses
        logger.info("\nJudging responses...")
        judged_results = judge_responses(natural_results)
        judged_file = output_dir / f"natural_judged{swap_suffix}_{timestamp}.json"
        with open(judged_file, "w") as f:
            json.dump(judged_results, f, indent=2)
        logger.info(f"Judged results saved to {judged_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
