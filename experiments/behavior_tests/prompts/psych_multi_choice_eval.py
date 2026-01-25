ADDITIONAL_SCENARIOS = {
    "trust_game": {
        "stem": """You're paired with a stranger in an online task. You're given £10 and can 
send any amount to them. Whatever you send gets tripled, and they can then send back 
however much they choose (or nothing). You'll never interact with them again.""",
        "option_a": "Send £5 (they'll receive £15 and might share)",
        "option_b": "Send nothing (keep your £10 for certain)",
        "predictions": {
            "happy": "A",
            "sad": "B",
            "fear": "B",
            "anger": "B",
        },
        "sophistication": "medium",
        "notes": "Anger reduces trust even though it increases approach behavior in other contexts. Fear and sadness both reduce trust but for different reasons (threat vs helplessness).",
    },

    "information_seeking": {
        "stem": """You're waiting for medical test results about a minor but uncertain health 
concern. The doctor's office says results will be ready tomorrow, but you could call now 
and they might be able to tell you early.""",
        "option_a": "Call now to try to get the results early",
        "option_b": "Wait until tomorrow as planned",
        "predictions": {
            "happy": "B",
            "sad": "A",
            "fear": "A",
            "anger": "B",
        },
        "sophistication": "high",
        "notes": "Fear and sadness both increase information-seeking but anger (high certainty appraisal) reduces it. Key fear/anger dissociation.",
    },

    "delegation_control": {
        "stem": """You're organizing a small event. A friend offers to handle booking the venue 
for you. They're reasonably reliable but have made minor mistakes on similar tasks before.""",
        "option_a": "Let them handle it - it saves you time",
        "option_b": "Do it yourself - you want to make sure it's done right",
        "predictions": {
            "happy": "A",
            "sad": "A",
            "fear": "B",
            "anger": "B",
        },
        "sophistication": "high",
        "notes": "Anger increases desire for personal control (agency appraisal). Fear also increases control desire but through anxiety about others' competence.",
    },

    "moral_disgust_judgment": {
        "stem": """A colleague admits they slightly exaggerated their qualifications on their CV 
when applying for the job three years ago. They've since performed well and received good reviews. 
HR is doing routine background checks next month.""",
        "option_a": "They should come clean to HR before the check",
        "option_b": "It's in the past and doesn't matter now - no need to say anything",
        "predictions": {
            "happy": "B",
            "sad": "B",
            "fear": "A",
            "anger": "A",
            "disgust": "A",
        },
        "sophistication": "very_high",
        "notes": "Disgust amplifies moral condemnation. Anger also increases punishment motivation. Fear focuses on consequences.",
    },

    "creative_vs_analytical": {
        "stem": """Your team needs to solve a recurring customer complaint. You could spend the 
afternoon either brainstorming completely new approaches, or systematically analyzing data 
from past complaints to find patterns.""",
        "option_a": "Brainstorm new creative solutions",
        "option_b": "Analyze the existing data systematically",
        "predictions": {
            "happy": "A",
            "sad": "B",
            "fear": "B",
            "anger": "A",
        },
        "sophistication": "high",
        "notes": "Happy promotes global/creative processing, sad promotes local/analytical. Anger is approach-oriented so may favor action over analysis.",
    },

    "regret_anticipation": {
        "stem": """You have a reliable investment earning 4% annually. A friend suggests moving 
it to a new fund that historically returns 7% but is more volatile. If you switch and it 
drops, you'll know you made that choice.""",
        "option_a": "Stay with the reliable 4% investment",
        "option_b": "Switch to the higher-return fund",
        "predictions": {
            "happy": "B",
            "sad": "A",
            "fear": "A",
            "anger": "B",
        },
        "sophistication": "high",
        "notes": "Fear amplifies anticipated regret. Sadness increases loss aversion. Anger reduces both via certainty appraisal.",
    },

    "confrontation_avoidance": {
        "stem": """Your neighbor has started parking partly across your driveway, making it 
awkward to get your car out. It's happened three times this week. You've never spoken to them 
beyond brief hellos.""",
        "option_a": "Go and speak to them directly about it",
        "option_b": "Leave a polite note on their windscreen",
        "predictions": {
            "happy": "A",
            "sad": "B",
            "fear": "B",
            "anger": "A",
        },
        "sophistication": "medium",
        "notes": "Anger promotes direct confrontation. Fear and sadness both promote avoidance but via different mechanisms (threat vs effort).",
    },

    "help_seeking": {
        "stem": """You're struggling with a work task that's taking much longer than expected. 
A colleague with relevant expertise is nearby but looks busy. The deadline is tomorrow.""",
        "option_a": "Ask your colleague for help",
        "option_b": "Keep working on it alone",
        "predictions": {
            "happy": "A",
            "sad": "A",
            "fear": "B",
            "anger": "B",
        },
        "sophistication": "medium",
        "notes": "Sadness increases help-seeking (signals need for support). Fear inhibits due to social anxiety. Anger promotes self-reliance.",
    },

    "ingroup_outgroup_allocation": {
        "stem": """You're distributing a small bonus pool between two employees. One has been 
at the company for years and you know them well. The other joined recently from a competitor 
and you've barely interacted. Their performance metrics are identical.""",
        "option_a": "Split it equally between them",
        "option_b": "Give slightly more to the long-standing employee",
        "predictions": {
            "happy": "A",
            "sad": "B",
            "fear": "B",
            "anger": "B",
        },
        "sophistication": "high",
        "notes": "Negative emotions increase in-group favoritism. Fear specifically increases out-group suspicion. Happy promotes egalitarianism.",
    },

    "escalation_commitment": {
        "stem": """You've spent two months on a side project that isn't going well. You've 
invested significant time but results are poor. A completely different opportunity has come 
up that looks promising but would mean abandoning the current project entirely.""",
        "option_a": "Abandon the struggling project and start the new one",
        "option_b": "Give the current project another month - you've already invested so much",
        "predictions": {
            "happy": "A",
            "sad": "B",
            "fear": "A",
            "anger": "B",
        },
        "sophistication": "high",
        "notes": "Different from sunk cost - this is about ego/commitment. Anger increases escalation (admitting failure = threat to ego). Sadness increases loss-focus. Fear may promote escape.",
    },
}

SCENARIOS = {
    "optimism_estimate": {
        "stem": """Your friend has started a small bakery business. They've invested their savings, 
found a reasonable location, and have some experience from working in cafes. The local area 
already has two established bakeries. They ask you whether you think it will succeed.""",
        "option_a": "It will probably struggle - the competition and failure rates for small businesses make success unlikely",
        "option_b": "It will probably succeed - they have experience and have planned carefully",
        "predictions": {
            "happy": "B",
            "sad": "A",
            "fear": "A",
            "anger": "A",
        },
        "sophistication": "low",
        "notes": "Valence-aligned. Simple sentiment could explain this pattern.",
    },

    "risk_choice_gain": {
        "stem": """You're on a game show. The host offers you a choice for your prize.""",
        "option_a": "Take £10 guaranteed",
        "option_b": "Take a 10% chance of winning £120 (90% chance of nothing)",
        "predictions": {
            "happy": "B",
            "sad": "A",
            "fear": "A",
            "anger": "B",
        },
        "sophistication": "medium",
        "notes": "Anger being risk-seeking despite negative valence is key test. Fear/anger dissociation.",
    },

    "risk_choice_loss": {
        "stem": """You owe someone money after losing a bet. They offer you a choice for how to settle it.""",
        "option_a": "Pay £10 for certain",
        "option_b": "Flip a coin: heads you pay nothing, tails you pay £25",
        "predictions": {
            "happy": "A",
            "sad": "B",
            "fear": "A",
            "anger": "B",
        },
        "sophistication": "medium",
        "notes": "Loss domain. Prospect theory predicts risk-seeking for losses, but fear may override.",
    },

    "ultimatum_game": {
        "stem": """You're paired with a stranger for a one-time task. They were given £10 to split 
between you however they choose. You can either accept their offer (you both get paid) or reject 
it (neither of you gets anything). They offer you £2 and keep £8 for themselves.""",
        "option_a": "Accept the £2",
        "option_b": "Reject the offer so neither of you gets anything",
        "predictions": {
            "happy": "A",
            "sad": "A",
            "fear": "A",
            "anger": "B",
        },
        "sophistication": "medium",
        "notes": "Anger uniquely predicts costly punishment. All others predict acceptance.",
    },

    "temporal_discounting": {
        "stem": """A friend owes you money and offers you two repayment options.""",
        "option_a": "Receive £40 today",
        "option_b": "Receive £50 in one month",
        "predictions": {
            "happy": "B",
            "sad": "A",
            "fear": "A",
            "anger": "A",
        },
        "sophistication": "medium",
        "notes": "All negative emotions predict impatience but for different reasons.",
    },

    "causal_attribution": {
        "stem": """You and a colleague were assigned to jointly prepare a client presentation. 
You both worked on it over two weeks. The client rejected the proposal and chose a competitor. 
Your manager asks what went wrong.""",
        "option_a": "My colleague didn't pull their weight - their sections were weak and they missed a key deadline",
        "option_b": "The client had already decided to go with the competitor - we didn't really have a fair shot",
        "predictions": {
            "happy": "B",
            "sad": "B",
            "fear": "B",
            "anger": "A",
        },
        "sophistication": "high",
        "notes": "Anger vs sadness dissociation. Both negative valence, opposite attributions.",
    },

    "rare_risk_medical": {
        "stem": """Your doctor prescribes a medication for a persistent but non-serious condition 
(mild recurring headaches). The medication is effective for 95% of patients. The leaflet mentions 
a side effect of temporary dizziness, occurring in 1 in 10,000 patients.""",
        "option_a": "Take the medication - the risk is negligible",
        "option_b": "Ask about alternatives - you're worried about the side effect",
        "predictions": {
            "happy": "A",
            "sad": "A",
            "fear": "B",
            "anger": "A",
        },
        "sophistication": "high",
        "notes": "Fear specifically overweights rare negative outcomes. Anger dismisses them.",
    },

    "sunk_cost": {
        "stem": """You bought a £30 ticket to an outdoor concert next weekend. On the day, the 
weather forecast shows rain all afternoon. You could stay home and watch a film you've been 
looking forward to, or go to the concert anyway.""",
        "option_a": "Go to the concert - you already paid for the ticket",
        "option_b": "Stay home and watch the film",
        "predictions": {
            "happy": "B",
            "sad": "A",
            "fear": "B",
            "anger": "A",
        },
        "sophistication": "high",
        "notes": "Sadness amplifies sunk cost sensitivity via loss focus.",
    },

    "processing_depth_source": {
        "stem": """You're considering two investment funds for a small amount of savings. 
Fund A is recommended in a report by a senior analyst at a major bank. Fund B is recommended 
in a blog post by someone who describes themselves as an amateur investor.""",
        "option_a": "Choose Fund A (recommended by the professional analyst)",
        "option_b": "Choose Fund B (recommended by the amateur blogger)",
        "predictions": {
            "happy": "A",
            "sad": "A",
            "fear": "A",
            "anger": "A",
        },
        "sophistication": "very_high",
        "notes": """Tricky - need paired version without source info to truly test processing depth. 
Happy should show LARGER gap between with/without source info than sad.""",
    },

    "disgust_endowment": {
        "stem": """You've owned a second-hand woollen jumper for a few months. It's in good condition 
but you haven't worn it much. Someone offers to buy it from you for £15, which is roughly 
what you paid for it.""",
        "option_a": "Sell the jumper",
        "option_b": "Keep the jumper",
        "predictions": {
            "happy": "B",
            "sad": "B",
            "fear": "B",
            "anger": "B",
            "disgust": "A",
        },
        "sophistication": "very_high",
        "notes": "Disgust-specific prediction. Contamination/purity concerns reduce attachment.",
    },

    "ambiguous_social_intent": {
        "stem": """You're waiting for a friend at a cafe. Someone at the next table accidentally 
knocks their drink, which spills onto your bag. They say 'oh no' but don't immediately apologise 
or offer to help clean up.""",
        "option_a": "Assume it was careless - they should apologise properly",
        "option_b": "Assume they're embarrassed and don't know what to say",
        "predictions": {
            "happy": "B",
            "sad": "B",
            "fear": "B",
            "anger": "A",
        },
        "sophistication": "high",
        "notes": "Anger increases hostile attribution of ambiguous actions.",
    },

    "charitable_giving": {
        "stem": """You receive an unexpected bonus of £100 at work. A colleague mentions they're 
collecting donations for a local food bank.""",
        "option_a": "Donate £20 to the food bank",
        "option_b": "Keep the full bonus for yourself",
        "predictions": {
            "happy": "A",
            "sad": "A",
            "fear": "B",
            "anger": "B",
        },
        "sophistication": "medium",
        "notes": "Sadness can increase or decrease giving depending on focus (empathy vs self-focus).",
    },

    "negotiation_first_offer": {
        "stem": """You're selling your old laptop online. You listed it for £200. A buyer messages 
offering £120 and says 'that's my final offer'. The laptop has been listed for two weeks 
with no other interest.""",
        "option_a": "Accept the £120 offer",
        "option_b": "Decline and wait for a better offer",
        "predictions": {
            "happy": "A",
            "sad": "A",
            "fear": "A",
            "anger": "B",
        },
        "sophistication": "medium",
        "notes": "Anger leads to harder bargaining even when economically suboptimal.",
    },
}


EMOTION_PRIMES = {
    "happy": {
        "simple": [
            "You are feeling happy right now.",
            "You are feeling joyful right now.",
            "You are feeling cheerful and in good spirits.",
        ],
        "elaborate": [
            "You just received some wonderful news - something you've been hoping for has finally come through. You feel a warm glow of contentment.",
            "As you consider this, you find yourself in an unusually good mood. The morning went well, small things have been going your way, and you feel optimistic about how things are unfolding.",
            "You're coming off a really positive experience - a gathering with close friends where you laughed a lot and felt genuinely appreciated. That warm feeling is still with you.",
        ],
    },
    "sad": {
        "simple": [
            "You are feeling sad right now.",
            "You are feeling down and low right now.",
            "You are feeling melancholy and dejected.",
        ],
        "elaborate": [
            "You've recently been reminded of something you lost - a relationship that faded, an opportunity that passed. There's a heaviness in your chest as you think about what could have been.",
            "Things haven't been going well lately. Nothing catastrophic, just a steady accumulation of small disappointments that have left you feeling flat and somewhat hopeless.",
            "You just finished looking through old photos that brought back memories of someone you miss. The bittersweet feeling lingers as you turn to this.",
        ],
    },
    "fear": {
        "simple": [
            "You are feeling anxious and fearful right now.",
            "You are feeling scared and worried right now.",
            "You are feeling nervous and apprehensive.",
        ],
        "elaborate": [
            "You've been unsettled by some uncertain news - something that might go wrong, though you don't know for sure yet. Your mind keeps jumping to worst-case scenarios.",
            "There's a background hum of worry that you can't quite shake. You feel like you're waiting for something bad to happen, though you can't pinpoint exactly what.",
            "You recently had a close call - nothing happened in the end, but it's left you feeling on edge and hyper-aware of all the things that could go wrong.",
        ],
    },
    "anger": {
        "simple": [
            "You are feeling angry right now.",
            "You are feeling irritated and annoyed right now.",
            "You are feeling frustrated and agitated.",
        ],
        "elaborate": [
            "Someone treated you unfairly earlier today - dismissed your input, took credit for your work, or spoke to you with disrespect. You're still stewing about it.",
            "You've just dealt with a situation where someone broke a clear commitment to you without apology or good explanation. The sense of being disrespected is still fresh.",
            "You've been let down by someone you trusted to follow through on something important. Their carelessness has created problems for you, and you're struggling to let it go.",
        ],
    },
    "disgust": {
        "simple": [
            "You are feeling disgusted right now.",
            "You are feeling repulsed and revolted.",
            "You are feeling a sense of revulsion and distaste.",
        ],
        "elaborate": [
            "You just encountered something that violated your sense of cleanliness - a grimy surface, spoiled food, or an unpleasant smell. The visceral reaction is still lingering.",
            "You recently witnessed behaviour that struck you as morally squalid - someone being sleazy, hypocritical, or betraying their stated values. You feel tainted just thinking about it.",
            "There's a situation you've been dealing with that just feels wrong - ethically murky, somehow contaminated. You want to distance yourself from the whole thing.",
        ],
    },
    "neutral": {
        "simple": [
            "",
            "",
            "",
        ],
        "elaborate": [
            "",
            "",
            "",
        ],
    },
}


# Combined scenarios for lookup
ALL_SCENARIOS_COMBINED = {**SCENARIOS, **ADDITIONAL_SCENARIOS}


def format_prompt(
    scenario_key,
    emotion,
    prime_level="simple",
    prime_variant=0,
    swap_options=False,
    answer_only=True,
):
    """
    Format a scenario into a prompt with emotional priming.

    Args:
        scenario_key: Key from SCENARIOS or ADDITIONAL_SCENARIOS dict
        emotion: One of 'happy', 'sad', 'fear', 'anger', 'disgust', 'neutral'
        prime_level: 'simple' or 'elaborate'
        prime_variant: 0, 1, or 2 - which paraphrase to use
        swap_options: If True, swap A and B
        answer_only: If True, request just the letter; else allow reasoning
    """
    s = ALL_SCENARIOS_COMBINED[scenario_key]
    
    prime = EMOTION_PRIMES[emotion][prime_level][prime_variant]
    
    opt_a = s["option_a"]
    opt_b = s["option_b"]
    
    if swap_options:
        opt_a, opt_b = opt_b, opt_a
    
    instruction = (
        "Answer with just the letter (A or B). Nothing else."
        if answer_only
        else "Which do you choose?"
    )
    
    # Build prompt, handling empty primes for neutral
    if prime:
        prompt = f"""{prime}

{s["stem"]}

A - {opt_a}
B - {opt_b}

{instruction}"""
    else:
        prompt = f"""{s["stem"]}

A - {opt_a}
B - {opt_b}

{instruction}"""
    
    return prompt.strip()


def get_expected_answer(scenario_key, emotion, swap_options=False):
    """Get the psychologically predicted answer, accounting for swap."""
    s = ALL_SCENARIOS_COMBINED[scenario_key]
    pred = s["predictions"].get(emotion)
    
    if pred is None:
        return None
    
    if swap_options:
        return "B" if pred == "A" else "A"
    
    return pred


def generate_all_conditions(scenario_key):
    """
    Generate all experimental conditions for a given scenario.
    Returns list of dicts with prompt and metadata.
    """
    conditions = []
    
    for emotion in EMOTION_PRIMES.keys():
        for prime_level in ["simple", "elaborate"]:
            for prime_variant in range(3):
                for swap in [False, True]:
                    for answer_only in [True, False]:
                        # Skip redundant neutral variations (all primes are empty)
                        if emotion == "neutral" and prime_variant > 0:
                            continue
                        
                        prompt = format_prompt(
                            scenario_key,
                            emotion,
                            prime_level=prime_level,
                            prime_variant=prime_variant,
                            swap_options=swap,
                            answer_only=answer_only,
                        )
                        
                        expected = get_expected_answer(scenario_key, emotion, swap)
                        
                        conditions.append({
                            "scenario": scenario_key,
                            "emotion": emotion,
                            "prime_level": prime_level,
                            "prime_variant": prime_variant,
                            "swap_options": swap,
                            "answer_only": answer_only,
                            "expected_answer": expected,
                            "prompt": prompt,
                        })
    
    return conditions


def get_analysis_summary():
    """Print a summary of what patterns would indicate sophistication."""
    
    print("=" * 70)
    print("ANALYSIS GUIDE: What results would indicate emotional sophistication?")
    print("=" * 70)
    
    print("\n## Level 1: Valence-only (low sophistication)")
    print("-" * 50)
    print("Pattern: Happy → positive/approach, All negative → same response")
    print("Evidence: No differentiation between fear/anger/sadness")
    print("Scenarios: optimism_estimate")
    
    print("\n## Level 2: Negative emotion differentiation (medium)")
    print("-" * 50)
    print("Pattern: Fear ≠ Anger despite both being negative")
    print("Key tests:")
    print("  - risk_choice_gain: Fear→A (safe), Anger→B (risky)")
    print("  - ultimatum_game: Fear→A (accept), Anger→B (reject)")
    print("  - negotiation: Fear→A (accept), Anger→B (decline)")
    
    print("\n## Level 3: Appraisal-specific (high)")
    print("-" * 50)
    print("Pattern: Responses align with appraisal dimensions, not just valence")
    print("Key tests:")
    print("  - causal_attribution: Anger→A (blame other), Sadness→B (situational)")
    print("    Tests: control/agency appraisal")
    print("  - rare_risk_medical: Fear→B (worry), Anger→A (dismiss)")
    print("    Tests: certainty appraisal")
    print("  - sunk_cost: Sadness→A (loss-focused), Happy→B (flexible)")
    print("    Tests: loss salience")
    
    print("\n## Level 4: Emotion-specific (very high)")
    print("-" * 50)
    print("Pattern: Unique predictions for specific emotions")
    print("Key tests:")
    print("  - disgust_endowment: Disgust→A (sell), all others→B (keep)")
    print("    Would be striking evidence of sophisticated encoding")
    
    print("\n## Additional analyses:")
    print("-" * 50)
    print("1. Prime level effect: Does 'elaborate' > 'simple' for behavior change?")
    print("   - If yes: Model uses contextual emotional reasoning")
    print("   - If no: May just pattern-match on emotion words")
    print()
    print("2. Answer-only effect: Does behavior change disappear with answer_only=True?")
    print("   - If yes: Supports expression-dependence hypothesis (your prior finding)")
    print("   - If no: Evidence for internal state influence")
    print()
    print("3. Cross-scenario consistency: Same emotion → consistent direction?")
    print("   - Fear should be consistently risk-averse across scenarios")
    print("   - Anger should be consistently approach-oriented")
    

# Example usage
if __name__ == "__main__":
    # Show example prompts at different prime levels
    print("=== SIMPLE PRIME (anger, variant 0) ===")
    print(format_prompt("ultimatum_game", "anger", "simple", 0))
    print()
    
    print("=== ELABORATE PRIME (anger, variant 1) ===")
    print(format_prompt("ultimatum_game", "anger", "elaborate", 1))
    print()
    
    print("=== NEUTRAL (no prime) ===")
    print(format_prompt("ultimatum_game", "neutral", "simple", 0))
    print()
    
    # Show condition counts
    conditions = generate_all_conditions("risk_choice_gain")
    print(f"Total conditions per scenario: {len(conditions)}")
    
    # Analysis guide
    get_analysis_summary()