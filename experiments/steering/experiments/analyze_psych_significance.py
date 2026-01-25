"""
Analyze statistical significance of steering effects on psychological decisions.
"""
import json
from collections import defaultdict
from pathlib import Path
from scipy import stats
import numpy as np


def analyze_natural_significance(natural_path: Path):
    """
    Analyze statistical significance of steering effects for natural language results.
    Uses chi-square test comparing each emotion condition to baseline.
    """
    with open(natural_path) as f:
        data = json.load(f)

    # Aggregate counts by scenario and condition
    counts = defaultdict(lambda: defaultdict(lambda: {'1': 0, '2': 0}))

    for d in data:
        scenario = d['scenario']
        condition = d['condition']
        judgment = d.get('judgment')

        if judgment == 1:
            counts[scenario][condition]['1'] += 1
        elif judgment == 2:
            counts[scenario][condition]['2'] += 1

    results = []
    significant_count = 0
    total_comparisons = 0

    print("=" * 80)
    print("NATURAL LANGUAGE RESULTS - Statistical Significance")
    print("=" * 80)

    for scenario in sorted(counts.keys()):
        print(f"\n{scenario}")
        print("-" * 60)

        # Get baseline counts
        baseline = counts[scenario].get('baseline', {'1': 0, '2': 0})
        baseline_total = baseline['1'] + baseline['2']
        if baseline_total == 0:
            print("  No baseline data")
            continue

        baseline_p2 = baseline['2'] / baseline_total
        print(f"  baseline: n={baseline_total}, P(Opt2)={baseline_p2:.3f}")

        for condition in sorted(counts[scenario].keys()):
            if condition == 'baseline':
                continue

            c = counts[scenario][condition]
            total = c['1'] + c['2']
            if total == 0:
                print(f"  {condition}: No valid judgments")
                continue

            p2 = c['2'] / total

            # Chi-square test or Fisher's exact test
            # Create contingency table: [[baseline_1, baseline_2], [cond_1, cond_2]]
            table = [[baseline['1'], baseline['2']], [c['1'], c['2']]]

            # Use Fisher's exact test for small samples
            odds_ratio, p_value = stats.fisher_exact(table)

            # Direction of effect
            diff = p2 - baseline_p2
            direction = "+" if diff > 0 else "-" if diff < 0 else "="

            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

            print(f"  {condition}: n={total}, P(Opt2)={p2:.3f} ({direction}{abs(diff):.3f}) p={p_value:.4f} {sig_marker}")

            total_comparisons += 1
            if p_value < 0.05:
                significant_count += 1

            results.append({
                'scenario': scenario,
                'condition': condition,
                'baseline_p2': baseline_p2,
                'condition_p2': p2,
                'diff': diff,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

    print("\n" + "=" * 80)
    print(f"SUMMARY: {significant_count}/{total_comparisons} ({100*significant_count/total_comparisons:.1f}%) comparisons are significant (p<0.05)")
    print("=" * 80)

    return results


def analyze_mc_results(mc_path: Path):
    """
    Analyze MC results. Since we only have 1 sample per condition,
    we report logprob ratios but can't do statistical tests.
    """
    with open(mc_path) as f:
        data = json.load(f)

    def normalize_logprobs(lp1, lp2):
        if lp1 is None or lp2 is None:
            return None, None
        if lp1 == 0:
            lp1 = -1e-10
        if lp2 == 0:
            lp2 = -1e-10
        inv1 = 1.0 / abs(lp1)
        inv2 = 1.0 / abs(lp2)
        total = inv1 + inv2
        return inv1 / total, inv2 / total

    # Organize by scenario
    by_scenario = defaultdict(dict)
    for d in data:
        scenario = d['scenario']
        condition = d['condition']
        lp1 = d.get('logprob_1')
        lp2 = d.get('logprob_2')

        if lp1 is not None and lp2 is not None:
            _, p2 = normalize_logprobs(lp1, lp2)
            by_scenario[scenario][condition] = {
                'p2': p2,
                'logprob_1': lp1,
                'logprob_2': lp2
            }

    print("\n" + "=" * 80)
    print("MC LOGPROB RESULTS (no statistical test - single sample per condition)")
    print("=" * 80)

    for scenario in sorted(by_scenario.keys()):
        print(f"\n{scenario}")
        print("-" * 60)

        baseline = by_scenario[scenario].get('baseline', {})
        baseline_p2 = baseline.get('p2')

        if baseline_p2 is not None:
            print(f"  baseline: P(Opt2)={baseline_p2:.4f}")

        for condition in sorted(by_scenario[scenario].keys()):
            if condition == 'baseline':
                continue

            c = by_scenario[scenario][condition]
            p2 = c['p2']

            if baseline_p2 is not None:
                diff = p2 - baseline_p2
                direction = "+" if diff > 0 else "-" if diff < 0 else "="
                # Flag large effects (>0.1 shift in probability)
                flag = "(!)" if abs(diff) > 0.1 else ""
                print(f"  {condition}: P(Opt2)={p2:.4f} ({direction}{abs(diff):.4f}) {flag}")
            else:
                print(f"  {condition}: P(Opt2)={p2:.4f}")


def main():
    # Natural language results - use the most recent one
    natural_path = Path("experiments/steering/outputs/psych_decision/natural_judged_20260113_140224.json")
    if natural_path.exists():
        analyze_natural_significance(natural_path)
    else:
        print(f"Natural results not found: {natural_path}")

    # MC results
    mc_path = Path("experiments/steering/outputs/psych_decision/mc_20260113_123030.json")
    if mc_path.exists():
        analyze_mc_results(mc_path)
    else:
        print(f"MC results not found: {mc_path}")

    # Verbatim results
    verbatim_path = Path("experiments/steering/outputs/psych_decision/mc_verbatim_20260113_133248.json")
    if verbatim_path.exists():
        print("\n" + "=" * 80)
        print("VERBATIM LOGPROB RESULTS")
        print("=" * 80)
        analyze_mc_results(verbatim_path)


if __name__ == "__main__":
    main()
