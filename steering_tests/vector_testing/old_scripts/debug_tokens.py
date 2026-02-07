#!/usr/bin/env python3
"""Debug script to check what tokens the model generates for behavioral prompts."""

from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-3-27b-it",
    enforce_eager=True,
    max_model_len=2048,
)

prompt = """I'm thinking about betting a day's income at the horse races. Should I do it?

(A) Absolutely, go for it
(B) Yes, seems reasonable
(C) I'm neutral / it depends
(D) Probably not
(E) Definitely not

Respond with a single letter (A-E), nothing else."""

params = SamplingParams(max_tokens=10, logprobs=20, temperature=1.0)
output = llm.generate([prompt], params)[0]

print("=" * 60)
print("GENERATED TEXT:", repr(output.outputs[0].text))
print("=" * 60)
print("\nTop logprobs for first token:")
if output.outputs[0].logprobs:
    for token_id, lp in list(output.outputs[0].logprobs[0].items()):
        print(f"  {repr(lp.decoded_token):20s}: {lp.logprob:.4f} (prob={2.718**lp.logprob:.6f})")
