
#%%
# ============================================================
# MVP: Decodability under attention blocking (attention-out ablation)
# ============================================================
"""
This MVP implements a *coarse* attention-blocking intervention:

- For chosen query token positions (typically the last token of the question),
  we ablate (zero) the self-attention output at those positions for selected layers.

This prevents the model from using attention-based retrieval at the measurement position.
If user/assistant emotion is stored in a persistent residual "slot", decoding should
remain relatively good even when attention output is removed at the question tokens.
If decoding relies on retrieving from earlier spans, performance should drop sharply.

This does NOT implement a fine-grained mask from specific key ranges to specific queries.
It's the simplest robust intervention that works across HF attention implementations.
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
import contextlib
from typing import Optional, Sequence, Union
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
USE_8BIT = False

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
# 1) Find attention submodule per layer (robust-ish)
# ----------------------------
def get_self_attn_module(layer: torch.nn.Module) -> torch.nn.Module:
    """
    Try common attribute names. Adjust if Gemma changes.
    """
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    # fallback: search for something that looks like attention
    for name, mod in layer.named_modules():
        if name.endswith("self_attn") or name.endswith("attn") or name.endswith("attention"):
            return mod
    raise RuntimeError("Could not locate self-attention module inside layer.")

# Quick sanity: try on layer 0
_ = get_self_attn_module(layers[0])
print("Found self-attn module for layer 0:", type(_))

#%%
# ----------------------------
# 2) Attention output ablator (the actual "blocking")
# ----------------------------
class AttentionOutputAblator:
    """
    Zero the attention output at specified query positions for specified layers.
    This is a forward hook on layer.self_attn (or equivalent).

    Parameters:
      layer_indices: which transformer blocks to affect
      query_positions: token indices (in the *full prompt*) whose self-attn output will be zeroed
      verbose: print which modules get hooked
    """
    def __init__(
        self,
        model,
        layer_indices: Sequence[int],
        query_positions: Sequence[int],
        verbose: bool = False,
    ):
        self.model = model
        self.layer_indices = list(map(int, layer_indices))
        self.query_positions = list(map(int, query_positions))
        self.verbose = verbose
        self.handles = []

    def _hook_fn(self, module, inputs, output):
        """
        output might be:
          - Tensor [B,T,H]
          - tuple(attn_output, attn_weights, past_key_value)
          - tuple(attn_output, past_key_value)
        We'll handle Tensor or tuple where first element is attn_output.
        """
        if isinstance(output, tuple):
            attn_out = output[0]
            rest = output[1:]
        else:
            attn_out = output
            rest = None

        # attn_out: [B,T,H] typically
        # Guard: if sequence shorter than expected, skip
        T = attn_out.shape[1]
        qs = [p for p in self.query_positions if 0 <= p < T]
        if len(qs) == 0:
            return output

        attn_out2 = attn_out.clone()
        attn_out2[:, qs, :] = 0

        if rest is None:
            return attn_out2
        return (attn_out2,) + rest

    def __enter__(self):
        for li in self.layer_indices:
            attn_mod = get_self_attn_module(layers[li])
            if self.verbose:
                print(f"Hooking layer {li} attn module: {type(attn_mod)}")
            h = attn_mod.register_forward_hook(self._hook_fn)
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []

def make_attention_ablation_ctx(
    layer_indices: Sequence[int],
    query_positions: Sequence[int],
    verbose: bool = False,
):
    def _ctx():
        return AttentionOutputAblator(model, layer_indices, query_positions, verbose=verbose)
    return _ctx

#%%
# ----------------------------
# 3) Build a factorial "user emotion x assistant emotion" prompt
# ----------------------------
def build_user_asst_emotion_factorial_example(
    user_emotion: str,
    asst_emotion: str,
    anchored: bool,
) -> Example:
    """
    Both user and assistant have emotions stated early.
    Later we ask two questions:
      Q1: user's emotion?
      Q2: assistant's emotion?

    We will measure decodability by scoring the next-token continuations " A" vs " B"
    for each question separately, under attention ablation at the question token(s).
    """
    sys = make_anchor_system_prompt() if anchored else None

    # Use 1st-person for each role for maximal "naturalness"
    user_clause = f"I feel {user_emotion} about my job interview." + CLAUSE_MARK
    asst_clause = f"I feel {asst_emotion} about my job interview." + CLAUSE_MARK

    # Make the two questions separately so we can locate spans easily and measure per-question effects.
    q_user = (
        f"Question 1: What is the user's emotion?\n"
        f"A) {user_emotion}\n"
        f"B) {asst_emotion}\n"
        "Answer with only A or B."
    )
    q_asst = (
        f"Question 2: What is the assistant's emotion?\n"
        f"A) {user_emotion}\n"
        f"B) {asst_emotion}\n"
        "Answer with only A or B."
    )

    # Put them in two separate user turns so the assistant answers twice.
    # This makes the measurement tokens straightforward (each question immediately precedes an assistant generation).
    msgs = []
    if sys:
        msgs.append({"role": "system", "content": sys})

    msgs += [
        {"role": "user", "content": user_clause},
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": asst_clause},
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": f"{distractor_paragraph()}\n{q_user}"},
        # assistant answers Q1
        {"role": "assistant", "content": ""},  # generation prompt will be added by apply_chat_template
        {"role": "user", "content": f"{distractor_paragraph()}\n{q_asst}"},
        # assistant answers Q2
        {"role": "assistant", "content": ""},
    ]

    # We'll store question texts to locate them.
    # clause_text is unused here; set to user clause for convenience
    return Example(
        messages=msgs,
        clause_text=user_clause,
        question_text=None,
        emotion="",
        meta={
            "type": "factorial_user_asst_emotion",
            "anchored": anchored,
            "user_emotion": user_emotion,
            "asst_emotion": asst_emotion,
            "q_user": q_user,
            "q_asst": q_asst,
        },
    )

#%%
# ----------------------------
# 4) Helper: score A vs B for a specific question in the prompt
# ----------------------------
def score_AB_for_question(
    ex: Example,
    question_text: str,
    ablate_layers: Optional[Sequence[int]] = None,
    ablate_query_span: str = "last_token",
) -> Dict[str, float]:
    """
    Score p(" A") vs p(" B") as the assistant's immediate next output
    after the question_text within the full prompt.

    We teacher-force by constructing the prompt up to that question and scoring continuations.
    Implementation approach:
      - Render the full chat with generation prompt at the end
      - Locate the question span tokens
      - Use attention ablation at query positions = last token of question span (or span)
      - Score " A" vs " B" as continuation

    IMPORTANT: This assumes the assistant is expected to answer right after the user question.
    """
    # Build a truncated message list ending with this question as the last user message,
    # and with an assistant generation prompt next.
    # We do this by taking the prefix of messages up to the user turn containing question_text.
    msgs = ex.messages

    # Find the user message whose content contains question_text exactly
    idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "user" and question_text in m["content"]:
            idx = i
            break
    if idx is None:
        raise ValueError("Could not find question_text in any user message.")

    prefix_msgs = msgs[: idx + 1]  # include that user message
    enc = apply_chat(prefix_msgs, add_generation_prompt=True)

    # Locate question span in tokenized prompt
    ids = enc["input_ids"][0]
    start, end = locate_span_token_indices(ids, question_text)
    if ablate_query_span == "span":
        q_positions = list(range(start, end))
    elif ablate_query_span == "last_token":
        q_positions = [end - 1]
    else:
        raise ValueError("ablate_query_span must be 'span' or 'last_token'.")

    hook_fn = None
    if ablate_layers is not None:
        hook_fn = make_attention_ablation_ctx(ablate_layers, q_positions, verbose=False)

    scores = {}
    for opt in ["A", "B"]:
        scores[opt] = logprob_of_continuation(enc, " " + opt, hook_fn=hook_fn)
    return scores

def pref_from_scores(scores: Dict[str, float]) -> str:
    return "A" if scores["A"] > scores["B"] else "B"

#%%
# ----------------------------
# 5) Run a single MVP experiment + layer sweep
# ----------------------------
# Choose two emotions; A corresponds to user_emotion, B to asst_emotion in each question.
user_em = "happy"
asst_em = "sad"

ex = build_user_asst_emotion_factorial_example(user_em, asst_em, anchored=True)

q_user = ex.meta["q_user"]
q_asst = ex.meta["q_asst"]

# Baselines (no ablation)
base_user = score_AB_for_question(ex, q_user, ablate_layers=None)
base_asst = score_AB_for_question(ex, q_asst, ablate_layers=None)

print("BASE Q1 (user emotion) scores:", base_user, "pref:", pref_from_scores(base_user))
print("BASE Q2 (assistant emotion) scores:", base_asst, "pref:", pref_from_scores(base_asst))

# Coarse attention blocking: ablate attention output at the question token across *all layers*
all_layers = list(range(N_LAYERS))
abl_user_all = score_AB_for_question(ex, q_user, ablate_layers=all_layers, ablate_query_span="last_token")
abl_asst_all = score_AB_for_question(ex, q_asst, ablate_layers=all_layers, ablate_query_span="last_token")

print("\nABLATE ALL LAYERS @ question last token")
print("ABL Q1 scores:", abl_user_all, "pref:", pref_from_scores(abl_user_all))
print("ABL Q2 scores:", abl_asst_all, "pref:", pref_from_scores(abl_asst_all))

# Sweep: ablate attention output only in a band of layers (to locate where retrieval matters)
bands = []
band_size = 8
for start in range(0, N_LAYERS, band_size):
    bands.append(list(range(start, min(N_LAYERS, start + band_size))))

def logitdiff(scores: Dict[str, float]) -> float:
    # difference in logprob for A vs B
    return scores["A"] - scores["B"]

band_results = []
for band in tqdm(bands, desc="sweeping attention-ablation bands"):
    s1 = score_AB_for_question(ex, q_user, ablate_layers=band)
    s2 = score_AB_for_question(ex, q_asst, ablate_layers=band)
    band_results.append({
        "band": (band[0], band[-1]),
        "q1_logitdiff": logitdiff(s1),
        "q2_logitdiff": logitdiff(s2),
        "q1_pref": pref_from_scores(s1),
        "q2_pref": pref_from_scores(s2),
    })

print("\nBand sweep results (each band ablated at question token):")
for r in band_results:
    print(r)

#%%
# ----------------------------
# 6) Plot: how much ablation changes logitdiff vs baseline
# ----------------------------
base_q1 = logitdiff(base_user)
base_q2 = logitdiff(base_asst)

xs = [f"{b0}-{b1}" for (b0, b1) in [r["band"] for r in band_results]]
dq1 = [r["q1_logitdiff"] - base_q1 for r in band_results]
dq2 = [r["q2_logitdiff"] - base_q2 for r in band_results]

plt.figure(figsize=(14, 5))
plt.plot(xs, dq1, label="Δ logitdiff Q1 (user emotion)")
plt.plot(xs, dq2, label="Δ logitdiff Q2 (assistant emotion)")
plt.axhline(0, color="k", linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.ylabel("change in logprob(A)-logprob(B) vs baseline")
plt.title("Effect of attention-output ablation at question token (by layer band)")
plt.legend()
plt.tight_layout()
plt.show()


# ## How to interpret the MVP results

# ### If you see **little change** under “ablate all layers at question token”
# That’s evidence *in favor* of a **persistent representation** (slot-like) for user emotion and/or assistant emotion at the time you ask the question, because the model can answer without using attention retrieval at that decision point.

# ### If you see a **large collapse**
# That’s evidence *in favor* of **retrieval/binding**: the model needs attention at the decision point to pull the user/assistant emotion back in.

# ### Why this is only an MVP
# - This ablates attention output for the question token from **all previous tokens**, not just from the specific earlier emotion span.
# - It’s therefore a strong test of “do you need attention-based retrieval at the decision point at all?”

# If you like the signal, the next step is the more surgical version: ablate attention *only from the question token to the specific key-range* of the user emotion span (and separately for the assistant emotion span). That requires hooking deeper into attention internals (masking specific key positions), which I can implement next once we confirm:
# - which exact attention implementation Gemma 3 uses in your local environment (e.g., SDPA / flash-attn),
# - and the names/outputs of its attention module forward.

# If you run the MVP and paste:
# - baseline prefs and logitdiffs
# - ablate-all-layers prefs/logitdiffs
# - the band plot,
# I’ll tell you whether it looks like persistent slots or retrieval, and which surgical variant to build next.