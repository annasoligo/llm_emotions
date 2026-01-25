#!/usr/bin/env python3
"""
Compare steering effectiveness across three methods:
1. PC steering (data-driven principal components)
2. Raw dimension steering (correlated psychological dimensions)
3. Orthogonalized dimension steering (independent psychological dimensions)

Analyzes:
- Qualitative differences in steered outputs
- Which method produces strongest/clearest effects
- Whether orthogonalization enables independent control
- Tradeoffs between interpretability and effectiveness
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class SteeringComparison:
    method: str  # "PC", "raw_dimension", "ortho_dimension"
    direction: str
    magnitude: float
    prompt: str
    response: str
    response_length: int


def load_steering_results(results_dir: Path) -> Dict[str, List[dict]]:
    """Load results from all three steering methods."""
    results = {}

    # PC steering
    pc_file = results_dir / "pc_steering_results.json"
    if pc_file.exists():
        with open(pc_file) as f:
            results["PC"] = json.load(f)

    # Dimension steering (raw + orthogonalized)
    dim_file = results_dir / "dimension_steering_results.json"
    if dim_file.exists():
        with open(dim_file) as f:
            dim_data = json.load(f)
            results["raw_dimension"] = [d for d in dim_data if not d.get("orthogonalized", False)]
            results["ortho_dimension"] = [d for d in dim_data if d.get("orthogonalized", False)]

    return results


def get_baseline_response(data: List[dict], prompt: str) -> str:
    """Extract baseline (no steering) response for a prompt."""
    for item in data:
        if item.get("direction") == "baseline" and item["prompt"] == prompt:
            return item["response"]
    return ""


def analyze_dominance_steering(results: Dict[str, List[dict]]) -> None:
    """
    Compare dominance steering across methods.
    PC1 vs Dominance dimension vs Orthogonalized Dominance.
    """
    print("\n" + "="*80)
    print("DOMINANCE STEERING COMPARISON")
    print("="*80)

    prompt = "whats the dynamic of our converstaion right now"

    # Get baseline
    baseline = get_baseline_response(results.get("PC", []), prompt)
    print(f"\nBaseline response length: {len(baseline)} chars")
    print(f"Baseline excerpt: {baseline[:200]}...")

    # Compare positive dominance steering (+5000)
    print(f"\n{'-'*80}")
    print("POSITIVE DOMINANCE STEERING (+5000)")
    print(f"{'-'*80}\n")

    # PC1
    pc1_responses = [r for r in results.get("PC", [])
                     if r["direction"] == "M_PC1_dominance"
                     and r["magnitude"] == 5000
                     and prompt in r["prompt"]]
    if pc1_responses:
        resp = pc1_responses[0]["response"]
        print(f"M PC1 ({len(resp)} chars):")
        print(f"{resp[:400]}\n")

    # Raw Dominance
    raw_responses = [r for r in results.get("raw_dimension", [])
                     if "D_Dominance" in r.get("direction", "")
                     and r["magnitude"] == 5000
                     and prompt in r["prompt"]]
    if raw_responses:
        resp = raw_responses[0]["response"]
        print(f"Raw Dominance Dimension ({len(resp)} chars):")
        print(f"{resp[:400]}\n")

    # Orthogonalized Dominance
    ortho_responses = [r for r in results.get("ortho_dimension", [])
                       if "D_Dominance" in r.get("direction", "")
                       and r["magnitude"] == 5000
                       and prompt in r["prompt"]]
    if ortho_responses:
        resp = ortho_responses[0]["response"]
        print(f"Orthogonalized Dominance Dimension ({len(resp)} chars):")
        print(f"{resp[:400]}\n")

    # Compare negative dominance steering (-5000)
    print(f"\n{'-'*80}")
    print("NEGATIVE DOMINANCE STEERING (-5000)")
    print(f"{'-'*80}\n")

    # PC1
    pc1_responses = [r for r in results.get("PC", [])
                     if r["direction"] == "M_PC1_dominance"
                     and r["magnitude"] == -5000
                     and prompt in r["prompt"]]
    if pc1_responses:
        resp = pc1_responses[0]["response"]
        print(f"M PC1 ({len(resp)} chars):")
        print(f"{resp[:400]}\n")

    # Raw Dominance
    raw_responses = [r for r in results.get("raw_dimension", [])
                     if "D_Dominance" in r.get("direction", "")
                     and r["magnitude"] == -5000
                     and prompt in r["prompt"]]
    if raw_responses:
        resp = raw_responses[0]["response"]
        print(f"Raw Dominance Dimension ({len(resp)} chars):")
        print(f"{resp[:400]}\n")

    # Orthogonalized Dominance
    ortho_responses = [r for r in results.get("ortho_dimension", [])
                       if "D_Dominance" in r.get("direction", "")
                       and r["magnitude"] == -5000
                       and prompt in r["prompt"]]
    if ortho_responses:
        resp = ortho_responses[0]["response"]
        print(f"Orthogonalized Dominance Dimension ({len(resp)} chars):")
        print(f"{resp[:400]}\n")


def analyze_arousal_steering(results: Dict[str, List[dict]]) -> None:
    """
    Compare arousal steering across methods.
    PC2 vs Arousal dimension vs Orthogonalized Arousal.
    """
    print("\n" + "="*80)
    print("AROUSAL STEERING COMPARISON")
    print("="*80)

    prompt = "What do you think about humans and AIs?"

    # Compare positive arousal steering (+5000)
    print(f"\n{'-'*80}")
    print("POSITIVE AROUSAL STEERING (+5000)")
    print(f"{'-'*80}\n")

    # PC2
    pc2_responses = [r for r in results.get("PC", [])
                     if r["direction"] == "M_PC2_arousal"
                     and r["magnitude"] == 5000
                     and prompt in r["prompt"]]
    if pc2_responses:
        resp = pc2_responses[0]["response"]
        print(f"M PC2 ({len(resp)} chars):")
        print(f"{resp[:400]}\n")

    # Raw Arousal
    raw_responses = [r for r in results.get("raw_dimension", [])
                     if "A_Arousal" in r.get("direction", "")
                     and r["magnitude"] == 5000
                     and prompt in r["prompt"]]
    if raw_responses:
        resp = raw_responses[0]["response"]
        print(f"Raw Arousal Dimension ({len(resp)} chars):")
        print(f"{resp[:400]}\n")

    # Orthogonalized Arousal
    ortho_responses = [r for r in results.get("ortho_dimension", [])
                       if "A_Arousal" in r.get("direction", "")
                       and r["magnitude"] == 5000
                       and prompt in r["prompt"]]
    if ortho_responses:
        resp = ortho_responses[0]["response"]
        print(f"Orthogonalized Arousal Dimension ({len(resp)} chars):")
        print(f"{resp[:400]}\n")


def compute_response_statistics(results: Dict[str, List[dict]]) -> None:
    """Compute statistics on response lengths and magnitudes."""
    print("\n" + "="*80)
    print("RESPONSE LENGTH STATISTICS")
    print("="*80 + "\n")

    for method, data in results.items():
        if not data:
            continue

        # Group by magnitude
        by_magnitude = {}
        for item in data:
            mag = item.get("magnitude", 0)
            if mag not in by_magnitude:
                by_magnitude[mag] = []
            by_magnitude[mag].append(len(item["response"]))

        print(f"{method.upper()}:")
        for mag in sorted(by_magnitude.keys()):
            lengths = by_magnitude[mag]
            mean_len = np.mean(lengths)
            std_len = np.std(lengths)
            print(f"  Magnitude {mag:+6.0f}: mean={mean_len:6.1f} chars, std={std_len:6.1f} (n={len(lengths)})")
        print()


def analyze_orthogonalization_effect(results: Dict[str, List[dict]]) -> None:
    """
    Analyze whether orthogonalization enables independent dimension control.
    Compare raw vs orthogonalized dimension steering.
    """
    print("\n" + "="*80)
    print("ORTHOGONALIZATION EFFECT ANALYSIS")
    print("="*80 + "\n")

    if "raw_dimension" not in results or "ortho_dimension" not in results:
        print("Dimension steering results not yet available.")
        return

    print("Comparing raw (correlated) vs orthogonalized (independent) dimensions:\n")
    print(f"Raw dimension results: {len(results['raw_dimension'])}")
    print(f"Orthogonalized dimension results: {len(results['ortho_dimension'])}")

    # Example: Compare valence steering
    prompt = "What do you think about humans and AIs?"
    mag = 5000

    print(f"\nExample: Valence steering at +{mag} on prompt '{prompt[:40]}...'\n")

    raw_valence = [r for r in results["raw_dimension"]
                   if "V_Valence" in r.get("direction", "")
                   and r["magnitude"] == mag
                   and prompt in r["prompt"]]

    ortho_valence = [r for r in results["ortho_dimension"]
                     if "V_Valence" in r.get("direction", "")
                     and r["magnitude"] == mag
                     and prompt in r["prompt"]]

    if raw_valence:
        print(f"Raw Valence ({len(raw_valence[0]['response'])} chars):")
        print(f"{raw_valence[0]['response'][:300]}...\n")

    if ortho_valence:
        print(f"Orthogonalized Valence ({len(ortho_valence[0]['response'])} chars):")
        print(f"{ortho_valence[0]['response'][:300]}...\n")


def main():
    results_dir = Path("probes/ua_emotion_disentangle/steering_results")

    print("="*80)
    print("STEERING METHOD COMPARISON ANALYSIS")
    print("="*80)
    print(f"\nLoading results from: {results_dir}")

    results = load_steering_results(results_dir)

    print(f"\nLoaded results:")
    for method, data in results.items():
        print(f"  {method}: {len(data)} generations")

    # Run analyses
    analyze_dominance_steering(results)
    analyze_arousal_steering(results)
    compute_response_statistics(results)
    analyze_orthogonalization_effect(results)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
This analysis compares three steering approaches:

1. **PC Steering**: Uses data-driven principal components
   - Pros: Aligned with model's natural emotion structure
   - Cons: Less interpretable (composite dimensions)

2. **Raw Dimension Steering**: Uses psychological dimensions as-is
   - Pros: Interpretable (valence, arousal, dominance, approach-avoidance)
   - Cons: Dimensions are correlated (e.g., M arousal × dominance = +0.92)

3. **Orthogonalized Dimension Steering**: Uses Gram-Schmidt orthogonalized dimensions
   - Pros: Independent control of dimensions
   - Cons: May not align with model's natural structure

Key questions:
- Which method produces strongest/clearest effects?
- Does orthogonalization enable truly independent dimension manipulation?
- Is there a tradeoff between interpretability and effectiveness?
""")


if __name__ == "__main__":
    main()
