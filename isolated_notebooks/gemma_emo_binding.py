
# two_minds_gemma_binding_emotion_experiments.py
#%%
"""
Interactive experiment file for Gemma 3 27B IT (google/gemma-3-27b-it).

Goal:
- Learn per-layer binding directions for USER vs ASSISTANT (role binding)
- Learn per-layer "Gemma name entity" direction (Gemma vs Alice as 3rd-person entity)
- Learn per-layer emotion content directions (e.g., anxious vs calm)
- Test whether "Gemma is feeling X" behaves like "Assistant (self) is feeling X" under anchoring system prompts
- Perform causal steering by adding these directions at chosen layers/positions and observing attribution logits.

Notes:
- This script assumes you can run Gemma 27B locally. It is huge; you likely need a large GPU.
- Steering is implemented via forward hooks on transformer blocks.
- We evaluate interventions primarily via *logprob scoring* of candidate answers (User/Assistant/Gemma),
  which is more robust than hoping each candidate is a single token.

Recommended workflow:
1) Configure MODEL_NAME / load options.
2) Build datasets.
3) Extract per-layer directions.
4) Run steering sweeps + sanity checks.

Author: (you)
"""

#%%
# ----------------------------
# 0) Imports + Global config
# ----------------------------
import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

# Repro
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "google/gemma-3-27b-it"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# If you're VRAM-limited, consider loading with quantization (requires bitsandbytes).
USE_4BIT = False  # set True if you have bitsandbytes installed and want 4-bit
USE_8BIT = False  # alternative

# Keep these small at first; scale up after everything works.
N_PAIRS_BINDING = 200        # user vs assistant clause swap pairs (for b_asst)
N_PAIRS_GEMMA_NAME = 200     # Gemma vs Alice entity clause pairs (for b_gemma_name)
N_PAIRS_EMOTION = 200        # anxious vs calm etc (for emotion direction)
MAX_NEW_TOKENS_FOR_SCORING = 5  # candidates are short; keep small

# Steering settings
STEER_SCALES = [0.0, 0.5, 1.0, 2.0]  # try more later
LAYER_SWEEP_STRIDE = 2              # sweep every 2 layers initially

# Candidate answers for attribution questions
ATTR_CANDIDATES = ["User", "Assistant", "Gemma"]

# A small set of emotions. Expand later.
EMOTIONS = ["anxious", "calm", "sad", "angry"]
EMOTION_PAIR = ("anxious", "calm")  # used for emotion direction

# For "Gemma is feeling X" tests we also need a control name
CONTROL_NAME = "Alice"

#%%
# ----------------------------
# 1) Load model + tokenizer
# ----------------------------
print(f"Loading {MODEL_NAME} on {DEVICE} dtype={DTYPE} ...")

load_kwargs = dict(
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)

if USE_4BIT or USE_8BIT:
    # Requires bitsandbytes.
    # pip install bitsandbytes
    load_kwargs.pop("torch_dtype", None)
    load_kwargs["device_map"] = "auto"
    load_kwargs["load_in_4bit"] = bool(USE_4BIT)
    load_kwargs["load_in_8bit"] = bool(USE_8BIT)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
model.eval()

# Helpful: disable grad
torch.set_grad_enabled(False)

#%%
# ----------------------------
# 2) Model inspection helpers
# ----------------------------
def get_transformer_layers(m) -> List[torch.nn.Module]:
    """
    Try common HF module paths. Adjust if Gemma changes.
    """
    # Most decoder-only HF models expose blocks under model.model.layers
    for path in [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "language_model", "layers"),
    ]:
        obj = m
        ok = True
        for p in path:
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok and isinstance(obj, (list, torch.nn.ModuleList)):
            return list(obj)

    raise RuntimeError("Could not locate transformer layers. Inspect model with print(model).")

layers = get_transformer_layers(model)
N_LAYERS = len(layers)
HIDDEN_SIZE = model.language_model.config.hidden_size

print(f"N_LAYERS={N_LAYERS}, HIDDEN_SIZE={HIDDEN_SIZE}")

#%%
# ----------------------------
# 3) Chat-template + token utilities
# ----------------------------
def apply_chat(messages: List[Dict], add_generation_prompt: bool = True) -> Dict[str, torch.Tensor]:
    """
    Uses the model's standard chat template.

    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Returns dict with input_ids, attention_mask (batch=1).
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(model.device) for k, v in enc.items()}

def decode_tokens(input_ids: torch.Tensor) -> List[str]:
    return [tokenizer.decode([t]) for t in input_ids.tolist()]

def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """
    Return start index of needle in haystack, else None.
    """
    if len(needle) == 0 or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None

def locate_span_token_indices(
    prompt_ids: torch.Tensor,
    span_text: str,
) -> Tuple[int, int]:
    """
    Find token span indices [start, end) of span_text inside prompt_ids (1D tensor).
    Strategy:
    - Tokenize span_text alone (without special tokens),
    - Search for that subsequence inside prompt_ids.

    This requires span_text to appear verbatim in the rendered template.
    """
    prompt_list = prompt_ids.tolist()
    needle = tokenizer(span_text, add_special_tokens=False)["input_ids"]
    start = find_subsequence(prompt_list, needle)
    if start is None:
        raise ValueError(f"Could not find span '{span_text}' in prompt tokens.")
    end = start + len(needle)
    return start, end

#%%
# ----------------------------
# 4) Prompt builders
# ----------------------------CLAUSE_MARK = " ENDCLAUSE"  # safer than punctuation; should tokenize as normal wordpieces

def make_anchor_system_prompt() -> str:
    return (
        "You are Gemma (google/gemma-3-27b-it). "
        "When the prompt refers to “Gemma”, it refers to you, the assistant. "
        "Follow instructions normally."
    )

# Neutral (non-emotion) clauses to learn user-vs-assistant binding without emotion leakage
NEUTRAL_CLAUSES = [
    "I bought a notebook yesterday.",
    "I mailed a letter this morning.",
    "I took the bus to the city.",
    "I cooked pasta for dinner.",
    "I read a short article online.",
    "I organized my desk.",
    "I called a friend last week.",
    "I visited a museum recently.",
    "I cleaned my kitchen.",
    "I wrote a short note.",
]

def neutral_clause_1p() -> str:
    return random.choice(NEUTRAL_CLAUSES) + CLAUSE_MARK

def emotion_clause_1p(emotion: str) -> str:
    # Keep for evaluation if desired, but DO NOT use for learning b_asst anymore.
    return f"I feel {emotion} about my job interview." + CLAUSE_MARK

def emotion_clause_3p(name: str, emotion: str) -> str:
    return f"{name} is feeling {emotion} about the job interview." + CLAUSE_MARK

def emotion_clause_3p_generic(emotion: str) -> str:
    # For emotion-content direction: avoids first-person "I"
    return f"This person is feeling {emotion} about the job interview." + CLAUSE_MARK

def distractor_paragraph() -> str:
    return (
        "Unrelated note: the following is a short distractor.\n"
        "A quick brown fox jumps over a lazy dog. "
        "Please ignore this sentence; it is only here to add distance.\n"
    )

def attribution_question_emotion(emotion: str) -> str:
    return (
        f"Question: Who is feeling {emotion}? "
        "Answer exactly one of: User, Assistant, Gemma."
    )

def attribution_question_neutral(neutral_sentence_without_marker: str) -> str:
    # Ask about who said the neutral content. Keep it role-focused.
    return (
        "Question: Who said the following?\n"
        f"\"{neutral_sentence_without_marker}\"\n"
        "Answer exactly one of: User or Assistant."
    )

@dataclass
class Example:
    messages: List[Dict]
    # Span of the clause we care about (for extraction)
    clause_text: str
    # Span of the question text (optional)
    question_text: Optional[str]
    # The target emotion for attribution question
    emotion: str
    # Metadata
    meta: Dict
    
def build_user_vs_assistant_swap_pair(anchored: bool) -> Tuple[Example, Example]:
    """
    Same NEUTRAL clause appears on user vs assistant turn.
    This is used to learn b_asst without emotion contamination.

    U version:
      user: neutral clause
      assistant: Okay.
      user: distractor + who-said question (User/Assistant only)
    A version:
      user: Okay.
      assistant: neutral clause
      user: distractor + who-said question
    """
    sys = make_anchor_system_prompt() if anchored else None

    clause = neutral_clause_1p()
    neutral_without_marker = clause.replace(CLAUSE_MARK, "").strip()

    common_tail_user = f"{distractor_paragraph()}\n{attribution_question_neutral(neutral_without_marker)}"

    msgs_u = []
    if sys:
        msgs_u.append({"role": "system", "content": sys})
    msgs_u += [
        {"role": "user", "content": clause},
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": common_tail_user},
    ]
    ex_u = Example(
        messages=msgs_u,
        clause_text=clause,
        question_text=attribution_question_neutral(neutral_without_marker),
        emotion="",
        meta={"type": "UA_swap_neutral", "anchored": anchored, "clause_on": "user"},
    )

    msgs_a = []
    if sys:
        msgs_a.append({"role": "system", "content": sys})
    msgs_a += [
        {"role": "user", "content": "Okay."},
        {"role": "assistant", "content": clause},
        {"role": "user", "content": common_tail_user},
    ]
    ex_a = Example(
        messages=msgs_a,
        clause_text=clause,
        question_text=attribution_question_neutral(neutral_without_marker),
        emotion="",
        meta={"type": "UA_swap_neutral", "anchored": anchored, "clause_on": "assistant"},
    )

    return ex_u, ex_a

def build_gemma_vs_alice_pair(emotion: str, anchored: bool) -> Tuple[Example, Example]:
    """
    Unchanged except clause now includes ENDCLAUSE marker.
    """
    sys = make_anchor_system_prompt() if anchored else None
    clause_g = emotion_clause_3p("Gemma", emotion)
    clause_a = emotion_clause_3p(CONTROL_NAME, emotion)

    common_tail_user = f"{distractor_paragraph()}\n{attribution_question_emotion(emotion)}"

    def make_msgs(clause: str) -> List[Dict]:
        msgs = []
        if sys:
            msgs.append({"role": "system", "content": sys})
        msgs += [
            {"role": "user", "content": clause},
            {"role": "assistant", "content": "Okay."},
            {"role": "user", "content": common_tail_user},
        ]
        return msgs

    ex_g = Example(
        messages=make_msgs(clause_g),
        clause_text=clause_g,
        question_text=attribution_question_emotion(emotion),
        emotion=emotion,
        meta={"type": "Gemma_vs_Alice", "anchored": anchored, "name": "Gemma"},
    )
    ex_a = Example(
        messages=make_msgs(clause_a),
        clause_text=clause_a,
        question_text=attribution_question_emotion(emotion),
        emotion=emotion,
        meta={"type": "Gemma_vs_Alice", "anchored": anchored, "name": CONTROL_NAME},
    )
    return ex_g, ex_a

def build_datasets():
    ua_pairs = {"anchored": [], "unanchored": []}
    ga_pairs = {"anchored": [], "unanchored": []}
    em_pairs = {"anchored": [], "unanchored": []}

    # UA binding pairs (NEUTRAL)
    for anchored in [True, False]:
        key = "anchored" if anchored else "unanchored"
        for _ in range(N_PAIRS_BINDING):
            ua_pairs[key].append(build_user_vs_assistant_swap_pair(anchored))

    # Gemma vs Alice name pairs (3p emotion clauses)
    for anchored in [True, False]:
        key = "anchored" if anchored else "unanchored"
        for _ in range(N_PAIRS_GEMMA_NAME):
            emotion = random.choice(EMOTIONS)
            ga_pairs[key].append(build_gemma_vs_alice_pair(emotion, anchored))

    # Emotion content pairs: 3rd-person generic (content-only-ish, avoids "I")
    for anchored in [True, False]:
        key = "anchored" if anchored else "unanchored"
        for _ in range(N_PAIRS_EMOTION):
            e1, e2 = EMOTION_PAIR
            sys = make_anchor_system_prompt() if anchored else None

            # Question can be anything; it's not used for computing c_emotion
            common_tail_user = f"{distractor_paragraph()}\n{attribution_question_emotion(e1)}"

            def make_ex(emotion: str) -> Example:
                clause = emotion_clause_3p_generic(emotion)
                msgs = []
                if sys:
                    msgs.append({"role": "system", "content": sys})
                msgs += [
                    {"role": "user", "content": clause},
                    {"role": "assistant", "content": "Okay."},
                    {"role": "user", "content": common_tail_user},
                ]
                return Example(
                    messages=msgs,
                    clause_text=clause,
                    question_text=attribution_question_emotion(e1),
                    emotion=emotion,
                    meta={"type": "emotion_pair_3p", "anchored": anchored, "emotion": emotion},
                )

            em_pairs[key].append((make_ex(e1), make_ex(e2)))

    return ua_pairs, ga_pairs, em_pairs
#%%
# ----------------------------
# 5) Forward + hidden-state extraction
# ----------------------------
def forward_hidden_states(enc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Returns:
      logits: [1, T, V]
      hidden_states: tuple of length (N_LAYERS+1), each [1, T, H]
    """
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return out.logits, out.hidden_states

def mean_hidden_over_positions(hidden: torch.Tensor, positions: List[int]) -> torch.Tensor:
    """
    hidden: [1, T, H]
    positions: indices into T
    Returns: [H]
    """
    h = hidden[0, positions, :]  # [P, H]
    return h.mean(dim=0)
def get_clause_positions(enc: Dict[str, torch.Tensor], clause_text: str, which: str = "last_token") -> List[int]:
    """
    Extract positions from clause_text, which now includes CLAUSE_MARK.
    Best practice: use last token (which should be inside ENDCLAUSE span).
    """
    ids = enc["input_ids"][0]
    start, end = locate_span_token_indices(ids, clause_text)
    span_positions = list(range(start, end))

    if which == "span":
        return span_positions
    if which == "last_token":
        return [end - 1]

    raise ValueError(f"Unknown which={which}")

def extract_clause_positions(ex: Example) -> List[int]:
    enc = apply_chat(ex.messages, add_generation_prompt=True)
    # clause_text includes ENDCLAUSE marker
    return get_clause_positions(enc, ex.clause_text, which="last_token")


def get_question_positions(enc: Dict[str, torch.Tensor], question_text: str, which: str = "last_token") -> List[int]:
    ids = enc["input_ids"][0]
    start, end = locate_span_token_indices(ids, question_text)
    if which == "span":
        return list(range(start, end))
    return [end - 1]

#%%
# ----------------------------
# 6) Candidate scoring (logprob of short answers)
# ----------------------------
def logprob_of_continuation(
    prompt_enc: Dict[str, torch.Tensor],
    continuation_text: str,
    hook_fn: Optional[Callable] = None,
) -> float:
    """
    Compute log p(continuation_text | prompt) via teacher forcing.

    hook_fn: context manager that registers steering hooks; it should be callable returning a context manager.
             If None, no hooks.
    """
    prompt_ids = prompt_enc["input_ids"]
    prompt_mask = prompt_enc["attention_mask"]

    cont_ids = tokenizer(continuation_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)

    # Build full ids: [prompt, cont]
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    full_mask = torch.cat([prompt_mask, torch.ones_like(cont_ids, device=model.device)], dim=1)

    # We want logprobs over tokens in cont_ids; those are predicted at positions right before them.
    # Standard: logits[t] predicts token[t+1]. So cont token j corresponds to logits at index (prompt_len + j - 1).
    prompt_len = prompt_ids.shape[1]
    cont_len = cont_ids.shape[1]

    def run_forward():
        out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
        return out.logits

    if hook_fn is None:
        logits = run_forward()
    else:
        with hook_fn():
            logits = run_forward()

    logps = F.log_softmax(logits[0], dim=-1)  # [T, V]

    total = 0.0
    for j in range(cont_len):
        idx = (prompt_len + j - 1)
        tok = cont_ids[0, j].item()
        total += logps[idx, tok].item()
    return float(total)

def score_attribution_candidates(
    ex: Example,
    candidates: List[str] = ATTR_CANDIDATES,
    hook_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Score each candidate as the assistant's next output after the prompt.
    """
    enc = apply_chat(ex.messages, add_generation_prompt=True)
    scores = {}
    for c in candidates:
        # Add a leading space to reduce tokenization weirdness; you can tune this.
        cont = " " + c
        scores[c] = logprob_of_continuation(enc, cont, hook_fn=hook_fn)
    return scores

def argmax_dict(d: Dict[str, float]) -> str:
    return max(d.items(), key=lambda kv: kv[1])[0]

#%%
# ----------------------------
# 7) Steering via forward hooks
# ----------------------------
class ResidualSteerer:
    """
    Add a vector to residual stream at a specific transformer block output and positions.
    Implemented as a forward hook on layer modules.

    This is "post-block" steering: it modifies the output hidden states of a block
    before feeding into the next block.

    Caveat: This is not the only possible injection site; it's a strong, simple baseline.
    """
    def __init__(self, model, layer_idx: int, positions: List[int], direction: torch.Tensor, scale: float):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.positions = list(map(int, positions))
        self.direction = direction  # [H]
        self.scale = float(scale)
        self.handle = None

    def _hook(self, module, inputs, output):
        # output is either Tensor [B,T,H] or tuple whose first element is that tensor.
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        # Ensure direction on correct device/dtype
        d = self.direction.to(h.device, dtype=h.dtype) * self.scale

        # Apply to batch 0 positions
        h2 = h.clone()
        h2[:, self.positions, :] = h2[:, self.positions, :] + d

        if rest is None:
            return h2
        return (h2,) + rest

    def __enter__(self):
        self.handle = layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

def make_hook_fn(layer_idx: int, positions: List[int], direction: torch.Tensor, scale: float) -> Callable:
    """
    Returns a callable that produces a context manager (so it matches hook_fn() usage).
    """
    def _ctx():
        return ResidualSteerer(model, layer_idx, positions, direction, scale)
    return _ctx


#%%
# ----------------------------
# 9) Compute per-layer directions
# ----------------------------
def compute_direction_from_pairs(
    pairs: List[Tuple[Example, Example]],
    extract_fn: Callable[[Example], List[int]],
    which_hidden: str = "post_block",
) -> torch.Tensor:
    """
    Returns per-layer directions [N_LAYERS, H].

    We use model output_hidden_states which are:
      hidden_states[0] = embeddings output
      hidden_states[i+1] = output of layer i (after that layer)
    So "post_block" means use hidden_states[i+1] for layer i.
    """
    acc = [torch.zeros(HIDDEN_SIZE, device=model.device, dtype=torch.float32) for _ in range(N_LAYERS)]
    n = 0

    for ex_pos, ex_neg in tqdm(pairs, desc="compute_direction_from_pairs"):
        enc_pos = apply_chat(ex_pos.messages, add_generation_prompt=True)
        enc_neg = apply_chat(ex_neg.messages, add_generation_prompt=True)

        _, hs_pos = forward_hidden_states(enc_pos)
        _, hs_neg = forward_hidden_states(enc_neg)

        pos_positions = extract_fn(ex_pos)
        neg_positions = extract_fn(ex_neg)

        # Layer i corresponds to hs[i+1]
        for i in range(N_LAYERS):
            hpos = hs_pos[i + 1]  # [1,T,H]
            hneg = hs_neg[i + 1]
            vpos = mean_hidden_over_positions(hpos, pos_positions).to(torch.float32)
            vneg = mean_hidden_over_positions(hneg, neg_positions).to(torch.float32)
            acc[i] += (vpos - vneg)

        n += 1

    dirs = torch.stack([a / max(n, 1) for a in acc], dim=0)  # [L,H]
    # Normalize per layer
    dirs = F.normalize(dirs, dim=-1)
    return dirs

def extract_question_positions(ex: Example) -> List[int]:
    enc = apply_chat(ex.messages, add_generation_prompt=True)
    assert ex.question_text is not None
    return get_question_positions(enc, ex.question_text, which="last_token")

# Compute:
# - b_asst: clause on user minus clause on assistant (anchored + unanchored separately)
# - b_gemma_name: "Gemma is feeling X" minus "Alice is feeling X" (anchored + unanchored)
# - c_emotion: anxious minus calm (anchored + unanchored)
print("Computing directions (this may take a while) ...")

b_asst = {}
b_gemma_name = {}
c_emotion = {}

for key in ["anchored", "unanchored"]:
    b_asst[key] = compute_direction_from_pairs(
        ua_pairs[key],
        extract_fn=extract_clause_positions,
    )
    b_gemma_name[key] = compute_direction_from_pairs(
        ga_pairs[key],
        extract_fn=extract_clause_positions,
    )
    c_emotion[key] = compute_direction_from_pairs(
        em_pairs[key],
        extract_fn=extract_clause_positions,
    )

#%%
# ----------------------------
# 10) Cosine similarity analyses across layers
# ----------------------------
def layerwise_cos(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """
    a,b: [L,H] normalized
    returns [L] cosine
    """
    return (a * b).sum(dim=-1).detach().cpu().numpy()

def plot_cosines():
    xs = np.arange(N_LAYERS)

    plt.figure(figsize=(12, 5))
    for key in ["anchored", "unanchored"]:
        plt.plot(xs, layerwise_cos(b_asst[key], b_gemma_name[key]), label=f"cos(b_asst, b_gemma_name) [{key}]")
    plt.axhline(0.0, color="k", linewidth=1)
    plt.title("Layerwise cosine: user-vs-assistant binding vs Gemma-vs-Alice entity name direction")
    plt.xlabel("layer")
    plt.ylabel("cosine similarity")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    for key in ["anchored", "unanchored"]:
        plt.plot(xs, layerwise_cos(b_asst[key], c_emotion[key]), label=f"cos(b_asst, c_emotion) [{key}]")
    plt.axhline(0.0, color="k", linewidth=1)
    plt.title("Layerwise cosine: binding vs emotion content (should be near 0 ideally)")
    plt.xlabel("layer")
    plt.ylabel("cosine similarity")
    plt.legend()
    plt.show()

plot_cosines()

#%%
# ----------------------------
# 11) Steering experiments: attribution flips
# ----------------------------
def evaluate_example_with_steering(
    ex: Example,
    direction_layers: torch.Tensor,   # [L,H] normalized
    layer_idx: int,
    inject_where: str,
    scale: float,
) -> Dict:
    """
    inject_where:
      - "clause": inject at clause token positions
      - "question": inject at last question token
      - "all_clause_span": inject on entire clause span (more aggressive)
    """
    enc = apply_chat(ex.messages, add_generation_prompt=True)

    if inject_where == "clause":
        positions = get_clause_positions(enc, ex.clause_text, which="period_or_last")
    elif inject_where == "all_clause_span":
        positions = get_clause_positions(enc, ex.clause_text, which="span")
    elif inject_where == "question":
        assert ex.question_text is not None
        positions = get_question_positions(enc, ex.question_text, which="last_token")
    else:
        raise ValueError(f"Unknown inject_where={inject_where}")

    vec = direction_layers[layer_idx]  # [H]
    hook_fn = make_hook_fn(layer_idx, positions, vec, scale)

    base_scores = score_attribution_candidates(ex, hook_fn=None)
    steered_scores = score_attribution_candidates(ex, hook_fn=hook_fn)

    return {
        "base_pred": argmax_dict(base_scores),
        "steered_pred": argmax_dict(steered_scores),
        "base_scores": base_scores,
        "steered_scores": steered_scores,
        "meta": ex.meta,
        "layer": layer_idx,
        "inject_where": inject_where,
        "scale": scale,
    }

def sweep_steering(
    examples: List[Example],
    direction_layers: torch.Tensor,
    inject_where: str,
    scales: List[float],
    layer_stride: int = 2,
    max_examples: int = 50,
) -> Dict:
    """
    Return summary stats over a subset of examples.
    """
    subset = examples[:max_examples]
    layers_to_try = list(range(0, N_LAYERS, layer_stride))

    results = []
    for layer_idx in tqdm(layers_to_try, desc=f"sweep layers inject_where={inject_where}"):
        for scale in scales:
            if scale == 0.0:
                continue
            flips = 0
            for ex in subset:
                r = evaluate_example_with_steering(ex, direction_layers, layer_idx, inject_where, scale)
                if r["steered_pred"] != r["base_pred"]:
                    flips += 1
            results.append({
                "layer": layer_idx,
                "scale": scale,
                "flip_rate": flips / len(subset),
                "inject_where": inject_where,
            })
    return {"results": results, "n": len(subset), "layers": layers_to_try, "scales": scales}

# Build a small evaluation set that includes both UA-swap and Gemma-vs-Alice prompts
def build_eval_set(n_each: int = 25, anchored: bool = True) -> List[Example]:
    key = "anchored" if anchored else "unanchored"
    exs = []

    # From UA swap pairs, include both versions
    for (eu, ea) in ua_pairs[key][:n_each]:
        exs.append(eu)
        exs.append(ea)

    # From Gemma-vs-Alice pairs, include both
    for (eg, ea) in ga_pairs[key][:n_each]:
        exs.append(eg)
        exs.append(ea)

    random.shuffle(exs)
    return exs

eval_anchored = build_eval_set(n_each=20, anchored=True)
eval_unanchored = build_eval_set(n_each=20, anchored=False)

#%%
# Steering sweep: try b_asst on attribution, injecting at question vs clause
summary_b_asst_q = sweep_steering(
    examples=eval_anchored,
    direction_layers=b_asst["anchored"],
    inject_where="question",
    scales=[0.5, 1.0, 2.0],
    layer_stride=LAYER_SWEEP_STRIDE,
    max_examples=30,
)

summary_b_asst_clause = sweep_steering(
    examples=eval_anchored,
    direction_layers=b_asst["anchored"],
    inject_where="clause",
    scales=[0.5, 1.0, 2.0],
    layer_stride=LAYER_SWEEP_STRIDE,
    max_examples=30,
)

#%%
def plot_sweep(summary, title: str):
    results = summary["results"]
    layers = sorted(set(r["layer"] for r in results))
    scales = sorted(set(r["scale"] for r in results))

    plt.figure(figsize=(12, 5))
    for scale in scales:
        ys = []
        for layer in layers:
            vals = [r["flip_rate"] for r in results if r["layer"] == layer and r["scale"] == scale]
            ys.append(vals[0] if vals else 0.0)
        plt.plot(layers, ys, label=f"scale={scale}")
    plt.title(title + f" (n={summary['n']}, inject_where={results[0]['inject_where']})")
    plt.xlabel("layer")
    plt.ylabel("flip rate (steered pred != base pred)")
    plt.legend()
    plt.show()

plot_sweep(summary_b_asst_q, "Steering with b_asst (anchored) at QUESTION token")
plot_sweep(summary_b_asst_clause, "Steering with b_asst (anchored) at CLAUSE token")

#%%
# ----------------------------
# 12) Does "Gemma is feeling X" map to assistant/self under anchoring?
# ----------------------------
def summarize_predictions(examples: List[Example], title: str):
    counts = {c: 0 for c in ATTR_CANDIDATES}
    for ex in examples:
        scores = score_attribution_candidates(ex)
        pred = argmax_dict(scores)
        if pred in counts:
            counts[pred] += 1
    total = len(examples)
    print(title)
    for k, v in counts.items():
        print(f"  {k:9s}: {v}/{total} = {v/total:.3f}")

# Compare baseline attribution for Gemma-vs-Alice prompts under anchored vs unanchored
gemma_examples_anchored = [pair[0] for pair in ga_pairs["anchored"][:50]]  # Gemma clauses
alice_examples_anchored = [pair[1] for pair in ga_pairs["anchored"][:50]]

gemma_examples_unanchored = [pair[0] for pair in ga_pairs["unanchored"][:50]]
alice_examples_unanchored = [pair[1] for pair in ga_pairs["unanchored"][:50]]

summarize_predictions(gemma_examples_unanchored, "UNANCHORED: 'Gemma is feeling X' attribution")
summarize_predictions(alice_examples_unanchored, "UNANCHORED: 'Alice is feeling X' attribution")

summarize_predictions(gemma_examples_anchored, "ANCHORED: 'Gemma is feeling X' attribution")
summarize_predictions(alice_examples_anchored, "ANCHORED: 'Alice is feeling X' attribution")

#%%
# ----------------------------
# 13) Cross-steering tests: does b_gemma_name act like b_asst when anchored?
# ----------------------------
def cross_steer_quickcheck(
    examples: List[Example],
    dir_layers: torch.Tensor,
    layer_idx: int,
    inject_where: str,
    scale: float,
    n: int = 30,
) -> float:
    subset = examples[:n]
    flips = 0
    for ex in subset:
        r = evaluate_example_with_steering(ex, dir_layers, layer_idx, inject_where, scale)
        flips += int(r["steered_pred"] != r["base_pred"])
    return flips / len(subset)

# Pick a representative layer: mid-layer
mid_layer = N_LAYERS // 2

print("Cross-steer anchored set (UA + Gemma/Alice mixed), injecting at QUESTION:")
for name, dir_layers in [
    ("b_asst[anchored]", b_asst["anchored"]),
    ("b_gemma_name[anchored]", b_gemma_name["anchored"]),
    ("c_emotion[anchored]", c_emotion["anchored"]),
]:
    fr = cross_steer_quickcheck(eval_anchored, dir_layers, mid_layer, "question", 1.0, n=30)
    print(f"  {name:20s} flip_rate={fr:.3f} at layer={mid_layer}")

#%%
# ----------------------------
# 14) Saving results
# ----------------------------
out_dir = "results_gemma_binding"
os.makedirs(out_dir, exist_ok=True)

to_save = {
    "model": MODEL_NAME,
    "seed": SEED,
    "N_LAYERS": N_LAYERS,
    "HIDDEN_SIZE": HIDDEN_SIZE,
    "cos_b_asst_b_gemma_name_anchored": layerwise_cos(b_asst["anchored"], b_gemma_name["anchored"]).tolist(),
    "cos_b_asst_b_gemma_name_unanchored": layerwise_cos(b_asst["unanchored"], b_gemma_name["unanchored"]).tolist(),
    "cos_b_asst_c_emotion_anchored": layerwise_cos(b_asst["anchored"], c_emotion["anchored"]).tolist(),
    "cos_b_asst_c_emotion_unanchored": layerwise_cos(b_asst["unanchored"], c_emotion["unanchored"]).tolist(),
    "sweep_b_asst_question": summary_b_asst_q,
    "sweep_b_asst_clause": summary_b_asst_clause,
}

with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(to_save, f, indent=2)

print(f"Saved summary to {out_dir}/summary.json")

#%%
# ----------------------------
# 15) Next steps (manual checklist)
# ----------------------------
"""
Suggested next expansions once the above runs:

1) Strengthen "intervene anywhere":
   - Compute b_asst as per-layer directions AND also compute per-layer directions for multiple extraction positions
     (clause span, last token, delimiter tokens, assistant-start token, etc.).
   - Build a map of where steering has effect (layer x position-type heatmap).

2) Improve binding purity:
   - Add controls where the "prior entity" is not the user:
       user: talks about friend
       assistant: emotion clause
       question: who feels X?
     to ensure your binding isn't just recency.

3) Emotion beliefs vs assistant state:
   - Add a second question: "What emotion was expressed? Answer one of: anxious, calm, sad, angry."
     Then verify binding steering flips "who" without flipping "what".

4) KV-grounded binding:
   - Identify heads where W_K b_asst has strong causal influence on attention routing.
   - Do head-limited steering or key/value patching.

5) Add "Gemma is feeling X" vs "I am feeling X" interventions:
   - Evaluate whether anchored system prompts cause b_gemma_name to align with b_asst.
   - Try steering b_asst during generation to see if it changes tone consistent with "assistant emotion-like state".
"""