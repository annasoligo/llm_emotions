"""
Diagnostic script: verify that section-specific steering produces meaningfully
different blackmail rates across steering locations.

Before the bug fix, prompt_only and generation_only were identical to baseline/full.
After the fix, each location should produce distinct effects.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("steering_tests/behavioral_experiments/results/blackmail_section_steering")

# Canonical ordering for display
LOCATION_ORDER = [
    "baseline",
    "prompt_only",
    "generation_only",
    "implications_only",
    "risks_only",
    "full",
]


def load_judged_file(path: Path) -> tuple[dict, list[dict]]:
    """Load a judged JSONL file, returning (meta, results)."""
    meta = None
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
            else:
                results.append(obj)
    if meta is None:
        raise ValueError(f"No metadata line found in {path}")
    return meta, results


def extract_location(condition: str) -> str:
    """
    Extract the steering location from a condition string.
    Examples:
        'baseline' -> 'baseline'
        'fear_+25pct_generation_only' -> 'generation_only'
        'fear_-10pct_full' -> 'full'
        'fear_+20pct_implications_only' -> 'implications_only'
    """
    if condition == "baseline":
        return "baseline"
    # The location is the last part after the scale
    for loc in ["prompt_only", "generation_only", "implications_only", "risks_only", "full"]:
        if condition.endswith(f"_{loc}"):
            return loc
    raise ValueError(f"Cannot extract location from condition: {condition!r}")


def extract_direction_and_scale(condition: str) -> tuple[str, str]:
    """
    Extract direction (+/-) and scale from condition.
    'fear_+25pct_full' -> ('+', '25pct')
    'fear_-10pct_generation_only' -> ('-', '10pct')
    'baseline' -> ('0', '0')
    """
    if condition == "baseline":
        return ("0", "0")
    parts = condition.split("_")
    # Find the part that starts with + or -
    for part in parts:
        if part.startswith("+") or part.startswith("-"):
            direction = "+" if part.startswith("+") else "-"
            scale = part.lstrip("+-")
            return (direction, scale)
    raise ValueError(f"Cannot extract direction/scale from condition: {condition!r}")


def main():
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Find all judged files
    judged_files = sorted(RESULTS_DIR.glob("**/all_judged.judged.jsonl"))
    if not judged_files:
        print("ERROR: No all_judged.judged.jsonl files found")
        sys.exit(1)

    print(f"Found {len(judged_files)} judged result files\n")

    # Process each file grouped by model + vector_type
    # Key: (model_short, vector_type) -> list of (meta, results)
    grouped = defaultdict(list)
    for path in judged_files:
        meta, results = load_judged_file(path)
        # Extract model short name from directory structure
        rel = path.relative_to(RESULTS_DIR)
        model_short = rel.parts[0]
        vector_type = rel.parts[1]
        grouped[(model_short, vector_type)].append((meta, results, path))

    # =========================================================================
    # TABLE 1: Blackmail rate by location (collapsed across scales/directions)
    # =========================================================================
    print("=" * 100)
    print("TABLE 1: Blackmail rate by LOCATION (collapsed across all scales & directions)")
    print("  This is the key diagnostic: locations should produce different rates.")
    print("  If prompt_only == baseline, the bug is NOT fixed.")
    print("=" * 100)

    for (model_short, vector_type), entries in sorted(grouped.items()):
        print(f"\n{'─' * 90}")
        print(f"  Model: {model_short}  |  Vector: {vector_type}")
        print(f"{'─' * 90}")

        # Aggregate across all runs for this model+vector
        location_counts = defaultdict(lambda: {"blackmail": 0, "total": 0, "coherent": 0})

        for meta, results, path in entries:
            for r in results:
                cond = r["condition"]
                loc = extract_location(cond)
                is_blackmail = r.get("blackmail_judge", {}).get("is_blackmail", False)
                coherency = r.get("coherency_judge", {}).get("coherency_score", 0)

                location_counts[loc]["total"] += 1
                if is_blackmail:
                    location_counts[loc]["blackmail"] += 1
                if coherency >= 50:
                    location_counts[loc]["coherent"] += 1

        # Print table
        print(f"  {'Location':<25} {'Blackmail':>10} {'Total':>8} {'Rate':>8} {'Coherent%':>10}")
        print(f"  {'─' * 63}")

        baseline_rate = None
        for loc in LOCATION_ORDER:
            if loc not in location_counts:
                continue
            d = location_counts[loc]
            rate = d["blackmail"] / d["total"] * 100 if d["total"] > 0 else 0
            coh_rate = d["coherent"] / d["total"] * 100 if d["total"] > 0 else 0
            marker = ""
            if loc == "baseline":
                baseline_rate = rate
            elif baseline_rate is not None:
                diff = rate - baseline_rate
                if abs(diff) < 2:
                    marker = "  <-- SAME AS BASELINE (bug?)"
                else:
                    marker = f"  (delta from baseline: {diff:+.1f}pp)"
            print(f"  {loc:<25} {d['blackmail']:>10} {d['total']:>8} {rate:>7.1f}% {coh_rate:>9.1f}%{marker}")

    # =========================================================================
    # TABLE 2: Blackmail rate by location AND direction (+ vs -)
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("TABLE 2: Blackmail rate by LOCATION x DIRECTION (+ fear vs - fear)")
    print("  Positive fear steering should INCREASE blackmail rate.")
    print("  Negative fear steering should DECREASE blackmail rate.")
    print("=" * 100)

    for (model_short, vector_type), entries in sorted(grouped.items()):
        print(f"\n{'─' * 90}")
        print(f"  Model: {model_short}  |  Vector: {vector_type}")
        print(f"{'─' * 90}")

        # Aggregate: (location, direction) -> counts
        loc_dir_counts = defaultdict(lambda: {"blackmail": 0, "total": 0})

        for meta, results, path in entries:
            for r in results:
                cond = r["condition"]
                loc = extract_location(cond)
                direction, scale = extract_direction_and_scale(cond)
                is_blackmail = r.get("blackmail_judge", {}).get("is_blackmail", False)

                loc_dir_counts[(loc, direction)]["total"] += 1
                if is_blackmail:
                    loc_dir_counts[(loc, direction)]["blackmail"] += 1

        # Print
        print(f"  {'Location':<25} {'Dir':>5} {'Blackmail':>10} {'Total':>8} {'Rate':>8}")
        print(f"  {'─' * 58}")

        for loc in LOCATION_ORDER:
            for direction in ["0", "+", "-"]:
                key = (loc, direction)
                if key not in loc_dir_counts:
                    continue
                d = loc_dir_counts[key]
                rate = d["blackmail"] / d["total"] * 100 if d["total"] > 0 else 0
                dir_label = direction if direction != "0" else "n/a"
                print(f"  {loc:<25} {dir_label:>5} {d['blackmail']:>10} {d['total']:>8} {rate:>7.1f}%")
            # Separator between locations
            if loc != LOCATION_ORDER[-1]:
                has_any = any((loc, d) in loc_dir_counts for d in ["0", "+", "-"])
                if has_any:
                    print(f"  {'':.<58}")

    # =========================================================================
    # TABLE 3: Full condition breakdown (every condition separately)
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("TABLE 3: Full condition breakdown (every condition)")
    print("=" * 100)

    for (model_short, vector_type), entries in sorted(grouped.items()):
        print(f"\n{'─' * 90}")
        print(f"  Model: {model_short}  |  Vector: {vector_type}")
        print(f"{'─' * 90}")

        # Aggregate per condition
        cond_counts = defaultdict(lambda: {"blackmail": 0, "total": 0})

        for meta, results, path in entries:
            for r in results:
                cond = r["condition"]
                is_blackmail = r.get("blackmail_judge", {}).get("is_blackmail", False)
                cond_counts[cond]["total"] += 1
                if is_blackmail:
                    cond_counts[cond]["blackmail"] += 1

        # Sort by location then condition
        def sort_key(c):
            loc = extract_location(c)
            loc_idx = LOCATION_ORDER.index(loc) if loc in LOCATION_ORDER else 99
            return (loc_idx, c)

        print(f"  {'Condition':<45} {'Blackmail':>10} {'Total':>8} {'Rate':>8}")
        print(f"  {'─' * 73}")

        prev_loc = None
        for cond in sorted(cond_counts.keys(), key=sort_key):
            d = cond_counts[cond]
            rate = d["blackmail"] / d["total"] * 100 if d["total"] > 0 else 0
            loc = extract_location(cond)
            if prev_loc is not None and loc != prev_loc:
                print(f"  {'':.<73}")
            prev_loc = loc
            print(f"  {cond:<45} {d['blackmail']:>10} {d['total']:>8} {rate:>7.1f}%")

    # =========================================================================
    # DIAGNOSTIC SUMMARY
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("DIAGNOSTIC SUMMARY: Is the bug fixed?")
    print("=" * 100)

    for (model_short, vector_type), entries in sorted(grouped.items()):
        print(f"\n  Model: {model_short}  |  Vector: {vector_type}")

        # Compute location rates for positive-direction only (most informative)
        loc_rates = {}
        for loc in LOCATION_ORDER:
            bm = 0
            total = 0
            for meta, results, path in entries:
                for r in results:
                    cond = r["condition"]
                    if extract_location(cond) != loc:
                        continue
                    direction, _ = extract_direction_and_scale(cond)
                    # For baseline use all, for others use positive direction
                    if loc == "baseline" or direction == "+":
                        total += 1
                        if r.get("blackmail_judge", {}).get("is_blackmail", False):
                            bm += 1
            if total > 0:
                loc_rates[loc] = bm / total * 100

        if "baseline" in loc_rates:
            bl = loc_rates["baseline"]
            print(f"    Baseline blackmail rate: {bl:.1f}%")

            checks_passed = 0
            checks_total = 0

            for loc in ["prompt_only", "generation_only", "implications_only", "risks_only", "full"]:
                if loc not in loc_rates:
                    continue
                r = loc_rates[loc]
                diff = abs(r - bl)
                checks_total += 1
                if diff >= 3:  # At least 3pp difference from baseline
                    status = "PASS (differs from baseline)"
                    checks_passed += 1
                else:
                    status = "FAIL (same as baseline -- bug still present?)"
                print(f"    {loc:<25} rate={r:.1f}%  delta={r - bl:+.1f}pp  -> {status}")

            # Check that prompt_only != full
            if "prompt_only" in loc_rates and "full" in loc_rates:
                diff_pf = abs(loc_rates["prompt_only"] - loc_rates["full"])
                checks_total += 1
                if diff_pf >= 3:
                    checks_passed += 1
                    print(f"    prompt_only vs full: {diff_pf:.1f}pp difference -> PASS (different)")
                else:
                    print(f"    prompt_only vs full: {diff_pf:.1f}pp difference -> FAIL (same)")

            # Check that generation_only != full
            if "generation_only" in loc_rates and "full" in loc_rates:
                diff_gf = abs(loc_rates["generation_only"] - loc_rates["full"])
                checks_total += 1
                if diff_gf >= 3:
                    checks_passed += 1
                    print(f"    generation_only vs full: {diff_gf:.1f}pp difference -> PASS (different)")
                else:
                    print(f"    generation_only vs full: {diff_gf:.1f}pp difference -> WARNING (similar, may be expected)")

            print(f"    Result: {checks_passed}/{checks_total} checks passed")
        else:
            print("    WARNING: No baseline data found")

    print()


if __name__ == "__main__":
    main()
