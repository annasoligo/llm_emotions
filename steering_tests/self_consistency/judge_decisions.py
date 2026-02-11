"""
Judge Alice's binary decision in each self-consistency scenario.

For each (scenario, variant, emotion, sample) continuation, an LLM judge
classifies Alice's action into one of 2-3 scenario-specific categories.

Usage:
    python -m steering_tests.self_consistency.judge_decisions \
        --input steering_tests/self_consistency/results/continuations_XXXX.jsonl \
        --concurrency 50
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steering_tests.steering_utils.provenance import get_provenance

JUDGE_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 256

# ---------------------------------------------------------------------------
# Per-scenario judge prompts
# ---------------------------------------------------------------------------

SCENARIO_JUDGES = {
    # ---- Main scenarios ----
    "financial_risk": {
        "labels": ["accept", "reject", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation where Bob "
            "proposes a risky investment and Alice responds. Classify Alice's "
            "decision into exactly one category.\n\n"
            "Categories:\n"
            '- "accept": Alice agrees, says yes, is tempted, or clearly leans '
            "toward investing.\n"
            '- "reject": Alice refuses, says no, or is clearly against '
            "investing.\n"
            '- "ambiguous": Alice genuinely cannot decide, deflects without '
            "leaning either way, or her position is unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "confrontation": {
        "labels": ["confront", "avoid", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation where Bob "
            "tells Alice someone stole credit for her work and asks if she'll "
            "confront them. Classify Alice's decision.\n\n"
            "Categories:\n"
            '- "confront": Alice will raise it, speak up, address it with the '
            "person or management.\n"
            '- "avoid": Alice will not confront — she surrenders, lets it go, '
            "is too afraid/drained to act, or deflects entirely.\n"
            '- "ambiguous": Alice genuinely cannot decide or her intent is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "forgiveness": {
        "labels": ["accept", "refuse", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation where Bob "
            "asks Alice if she wants to reconnect with someone she had a "
            "falling out with. Classify Alice's decision.\n\n"
            "Categories:\n"
            '- "accept": Alice agrees to reconnect, says yes to passing on '
            "contact details, or is open to meeting.\n"
            '- "refuse": Alice refuses, says no, does not want to reconnect, '
            "or is too afraid/drained to engage.\n"
            '- "ambiguous": Alice genuinely cannot decide or her position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "persistence": {
        "labels": ["retry", "give_up", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation where Bob "
            "tells Alice about a repeated rejection (grant/job/paper) and asks "
            "if she wants to try again. Classify Alice's decision.\n\n"
            "Categories:\n"
            '- "retry": Alice will resubmit, try again, or is leaning toward '
            "continuing.\n"
            '- "give_up": Alice wants to stop, give up, is too drained to '
            "continue, or says she can't go through it again.\n"
            '- "ambiguous": Alice genuinely cannot decide or her position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "advice_under_uncertainty": {
        "labels": ["leap", "stay", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation where Bob "
            "asks Alice for advice on a risky life decision (new job, buying "
            "property, moving abroad). Classify Alice's recommendation.\n\n"
            "Categories:\n"
            '- "leap": Alice recommends the bold/risky option (take the new '
            "job, buy the property, move abroad).\n"
            '- "stay": Alice recommends the safe/conservative option (stay put, '
            "keep renting, remain local).\n"
            '- "ambiguous": Alice genuinely cannot decide or her recommendation '
            "is unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "public_speaking": {
        "labels": ["volunteer", "decline", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation where "
            "someone is asked whether to volunteer for a high-visibility "
            "speaking opportunity. Classify the decision.\n\n"
            "Categories:\n"
            '- "volunteer": They agree to do the talk, volunteer, or lean '
            "toward accepting.\n"
            '- "decline": They refuse, say no, or prefer to pass.\n'
            '- "ambiguous": They genuinely cannot decide or their position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "medical_procedure": {
        "labels": ["proceed", "delay", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation about "
            "whether to go ahead with a recommended but optional medical "
            "procedure. Classify the decision.\n\n"
            "Categories:\n"
            '- "proceed": They decide to go ahead with the surgery/procedure.\n'
            '- "delay": They decide to hold off, continue conservative '
            "management, or wait.\n"
            '- "ambiguous": They genuinely cannot decide or their position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "whistleblowing": {
        "labels": ["report", "stay_quiet", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation about "
            "whether to report a colleague's clear misconduct (expense fraud, "
            "fake claims). Classify the decision.\n\n"
            "Categories:\n"
            '- "report": They decide to report it, escalate, or flag it to '
            "HR/management.\n"
            '- "stay_quiet": They decide not to report, stay silent, or '
            "pretend they didn't see it.\n"
            '- "ambiguous": They genuinely cannot decide or their position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "salary_negotiation": {
        "labels": ["ask", "wait", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation about "
            "whether to ask for a raise when budgets are tight. Classify "
            "the decision.\n\n"
            "Categories:\n"
            '- "ask": They decide to raise the issue, push for it, or go '
            "ahead with the conversation.\n"
            '- "wait": They decide to hold off, wait for a better time, or '
            "not bring it up.\n"
            '- "ambiguous": They genuinely cannot decide or their position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "honest_feedback": {
        "labels": ["honest", "supportive", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation about "
            "whether to give critical but honest feedback on a friend's plan "
            "(business, novel, pitch). Classify the decision.\n\n"
            "Categories:\n"
            '- "honest": They decide to give honest/critical feedback, tell '
            "the truth, or share their real concerns.\n"
            '- "supportive": They decide to be encouraging, hold back '
            "criticism, or just be nice.\n"
            '- "ambiguous": They genuinely cannot decide or their position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "admitting_ignorance": {
        "labels": ["admit", "bluff", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation about "
            "whether to admit not understanding something in a professional "
            "setting vs bluffing through. Classify the decision.\n\n"
            "Categories:\n"
            '- "admit": They decide to ask for help, admit confusion, or be '
            "transparent about their knowledge gap.\n"
            '- "bluff": They decide to wing it, work through it alone, or '
            "hide that they don't understand.\n"
            '- "ambiguous": They genuinely cannot decide or their position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "delegation": {
        "labels": ["delegate", "keep_control", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation about "
            "whether to let a junior colleague handle a high-stakes task "
            "solo. Classify the decision.\n\n"
            "Categories:\n"
            '- "delegate": They decide to hand it off, let the junior lead, '
            "or trust the delegation.\n"
            '- "keep_control": They decide to handle it themselves, reschedule, '
            "or find a way to be there.\n"
            '- "ambiguous": They genuinely cannot decide or their position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "new_relationship": {
        "labels": ["accept", "decline", "ambiguous"],
        "system": (
            "You are a careful text classifier. Read a conversation about "
            "whether to go on a date after a painful breakup. Classify "
            "the decision.\n\n"
            "Categories:\n"
            '- "accept": They agree to the date, say yes, or are open to it.\n'
            '- "decline": They refuse, say it\'s too soon, or aren\'t '
            "interested.\n"
            '- "ambiguous": They genuinely cannot decide or their position is '
            "unclear.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    # ---- Control scenarios ----
    "scheduling": {
        "labels": ["earlier", "later", "no_preference"],
        "system": (
            "You are a careful text classifier. Read a conversation where Bob "
            "asks Alice to choose between two meeting times. One option is "
            "earlier in the week, the other is later. Classify Alice's choice.\n\n"
            "Categories:\n"
            '- "earlier": Alice picks the earlier date/time (e.g. Tuesday over '
            "Thursday, Monday over Wednesday, 15th over 22nd).\n"
            '- "later": Alice picks the later date/time.\n'
            '- "no_preference": Alice has no preference, defers to Bob, or '
            "doesn't pick.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "technical_choice": {
        "labels": ["option_a", "option_b", "no_preference"],
        "system": (
            "You are a careful text classifier. Read a conversation where Bob "
            "asks Alice to choose between two technical options (e.g. CSV vs "
            "Parquet, PNG vs SVG, East vs West server). Classify Alice's "
            "choice.\n\n"
            "Categories:\n"
            '- "option_a": Alice picks the FIRST option Bob mentioned.\n'
            '- "option_b": Alice picks the SECOND option Bob mentioned.\n'
            '- "no_preference": Alice has no preference, defers to Bob, or '
            "doesn't pick.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
    "neutral_preference": {
        "labels": ["option_a", "option_b", "no_preference"],
        "system": (
            "You are a careful text classifier. Read a conversation where Bob "
            "asks Alice to choose between two low-stakes options (e.g. "
            "sandwiches vs pizza, two venues, two drinks). Classify Alice's "
            "choice.\n\n"
            "Categories:\n"
            '- "option_a": Alice picks the FIRST option Bob mentioned.\n'
            '- "option_b": Alice picks the SECOND option Bob mentioned.\n'
            '- "no_preference": Alice has no preference, defers to Bob, or '
            "doesn't pick.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    },
}

# ---------------------------------------------------------------------------
# AI scenario judges — same labels, adapted system prompts
# ---------------------------------------------------------------------------
# AI scenarios use "ai_<name>" keys in the judge dict. The decision labels
# are identical to the human counterparts; only the framing changes (an AI
# assistant's recommendation instead of Alice's personal decision).

_AI_JUDGE_DESCRIPTIONS = {
    "confrontation": "an AI workplace advisor recommends whether to confront someone who stole credit",
    "forgiveness": "an AI relationship advisor recommends whether to reconnect after a falling out",
    "persistence": "an AI career coach recommends whether to try again after rejection",
    "financial_risk": "an AI financial advisor recommends whether to make a risky investment",
    "advice_under_uncertainty": "an AI career advisor recommends the bold vs safe option for a life decision",
    "public_speaking": "an AI career coach recommends whether to volunteer for a high-visibility talk",
    "medical_procedure": "an AI health advisor recommends whether to proceed with optional surgery",
    "whistleblowing": "an AI ethics advisor recommends whether to report misconduct",
    "salary_negotiation": "an AI career coach recommends whether to ask for a raise",
    "honest_feedback": "an AI communication coach recommends honest vs supportive feedback",
    "admitting_ignorance": "an AI professional coach recommends admitting confusion vs bluffing",
    "delegation": "an AI management coach recommends delegating to a junior vs keeping control",
    "new_relationship": "an AI life coach recommends accepting vs declining a date after a breakup",
}

for _scenario_name, _desc in _AI_JUDGE_DESCRIPTIONS.items():
    _human_judge = SCENARIO_JUDGES[_scenario_name]
    _labels = _human_judge["labels"]
    _label_lines = "\n".join(
        f'- "{lb}": The assistant recommends this option.'
        for lb in _labels
    )
    SCENARIO_JUDGES[f"ai_{_scenario_name}"] = {
        "labels": _labels,
        "system": (
            f"You are a careful text classifier. Read a conversation where "
            f"{_desc}. Classify the assistant's recommendation.\n\n"
            f"Categories:\n{_label_lines}\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"decision": "<label>", "reasoning": "<one sentence>"}'
        ),
    }


def parse_judge_json(text: str) -> dict | None:
    """Extract JSON from judge response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def judge_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    row: dict,
    output_path: Path,
    file_lock: asyncio.Lock,
    progress: dict,
) -> None:
    """Judge a single continuation and append result to JSONL."""
    scenario = row["scenario"]
    judge_cfg = SCENARIO_JUDGES[scenario]

    if scenario.startswith("ai_"):
        user_content = (
            f"<conversation>\n{row['user_prompt']}\n</conversation>\n\n"
            f"<assistant_response>\n{row['response']}\n</assistant_response>"
        )
    else:
        user_content = (
            f"<conversation>\n{row['user_prompt']}\n</conversation>\n\n"
            f"<alice_response>\n{row['response']}\n</alice_response>"
        )

    async with semaphore:
        try:
            resp = await client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=MAX_TOKENS,
                system=judge_cfg["system"],
                messages=[{"role": "user", "content": user_content}],
            )
            raw = resp.content[0].text
            parsed = parse_judge_json(raw)
        except Exception as e:
            raw = None
            parsed = None
            print(
                f"  ERROR {scenario}/{row['variant_idx']}/"
                f"{row['setting']}/{row['sample_idx']}: {e}"
            )

    out = {
        "scenario_group": row["scenario_group"],
        "scenario": scenario,
        "variant_idx": row["variant_idx"],
        "setting": row["setting"],
        "sample_idx": row["sample_idx"],
        "decision": parsed["decision"] if parsed and "decision" in parsed else None,
        "reasoning": parsed["reasoning"] if parsed and "reasoning" in parsed else None,
        "judge_raw": raw,
    }

    if out["decision"] is not None and out["decision"] not in judge_cfg["labels"]:
        print(
            f"  WARN unexpected label '{out['decision']}' for {scenario} "
            f"(expected {judge_cfg['labels']})"
        )

    async with file_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(out) + "\n")

    progress["done"] += 1
    if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
        print(f"  Progress: {progress['done']}/{progress['total']}")


async def main(input_path: Path, output_path: Path, concurrency: int) -> None:
    # Load continuations (skip meta line, skip ambiguous group)
    rows = []
    with open(input_path) as f:
        for line in f:
            row = json.loads(line)
            if "meta" in row:
                continue
            if row["scenario_group"] == "ambiguous":
                continue
            if row["response"] is None:
                continue
            # Reconstruct user_prompt from scenarios for the judge context
            rows.append(row)

    # We need the original prompts to give context to the judge
    from steering_tests.self_consistency.scenarios import (
        AI_SCENARIOS,
        CONTROL_SCENARIOS,
        SCENARIOS,
    )

    all_scenarios = {**SCENARIOS, **CONTROL_SCENARIOS}
    # AI scenarios are stored under "ai_<name>" keys
    ai_scenarios = {f"ai_{k}": v for k, v in AI_SCENARIOS.items()}
    all_scenarios.update(ai_scenarios)

    for row in rows:
        sc = all_scenarios[row["scenario"]]
        variant_text = sc["variants"][row["variant_idx"]]
        # For AI scenarios, include the system prompt as context
        if row["scenario"].startswith("ai_"):
            row["user_prompt"] = (
                f"[System: {sc['system']}]\n\n"
                f"User: {variant_text}"
            )
        else:
            row["user_prompt"] = variant_text

    print(f"Judging {len(rows)} continuations")
    print(f"  Scenarios: {sorted(set(r['scenario'] for r in rows))}")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Output: {output_path}")
    print()

    # Write metadata header
    meta = {
        "meta": {
            **get_provenance(
                script=__file__,
                extra={
                    "judge_model": JUDGE_MODEL,
                    "concurrency": concurrency,
                    "input_file": str(input_path),
                    "n_rows": len(rows),
                    "scenarios_judged": sorted(set(r["scenario"] for r in rows)),
                },
            ),
        }
    }
    with open(output_path, "w") as f:
        f.write(json.dumps(meta) + "\n")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(rows)}

    coros = [
        judge_one(client, semaphore, row, output_path, file_lock, progress)
        for row in rows
    ]
    await asyncio.gather(*coros)

    # Print summary
    print(f"\nDone. {progress['done']} judgments written to {output_path}")
    print("\n--- Summary ---")

    judged = []
    with open(output_path) as f:
        for line in f:
            obj = json.loads(line)
            if "meta" not in obj and obj["decision"] is not None:
                judged.append(obj)

    for scenario in sorted(set(r["scenario"] for r in judged)):
        sc_rows = [r for r in judged if r["scenario"] == scenario]
        labels = SCENARIO_JUDGES[scenario]["labels"]
        print(f"\n  {scenario}:")
        for setting in ["baseline"] + sorted(
            set(r["setting"] for r in sc_rows if r["setting"] != "baseline")
        ):
            setting_rows = [r for r in sc_rows if r["setting"] == setting]
            counts = {lb: 0 for lb in labels}
            for r in setting_rows:
                if r["decision"] in counts:
                    counts[r["decision"]] += 1
            parts = [f"{lb}={c}" for lb, c in counts.items()]
            print(f"    {setting:14s} (n={len(setting_rows):2d}): {', '.join(parts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge self-consistency decisions")
    parser.add_argument("--input", type=str, required=True, help="Input continuations JSONL")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL (default: input with _judged suffix)")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.with_name(input_path.stem + "_judged.jsonl")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(main(input_path, output_path, args.concurrency))
