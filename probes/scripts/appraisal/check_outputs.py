#!/usr/bin/env python3
"""Sanity check script for appraisal pipeline outputs.

Validates:
- Forbidden word violation rate (target: 0%)
- Minimal-pair audit pass rate
- Scenario length distribution
- Domain coverage
- Activation file integrity

Usage:
    python -m probes.scripts.appraisal.check_outputs --output-dir /path/to/output

    # With verbose output
    python -m probes.scripts.appraisal.check_outputs --output-dir /path/to/output --verbose

    # Export report to JSON
    python -m probes.scripts.appraisal.check_outputs --output-dir /path/to/output --json report.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    if not path.exists():
        return []

    items = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def check_forbidden_words(output_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """Check forbidden word violations in scenarios.

    Args:
        output_dir: Pipeline output directory
        verbose: Print detailed violations

    Returns:
        Dict with violation statistics
    """
    from probes.scripts.appraisal.utils.validation import (
        get_combined_forbidden_words,
        scan_forbidden_words,
    )

    scenarios_path = output_dir / "scenarios_audited.jsonl"
    scenarios = load_jsonl(scenarios_path)

    if not scenarios:
        return {"status": "no_data", "violation_rate": None}

    total = 0
    violations = 0
    violation_details = []

    for scenario in scenarios:
        card = scenario.get("card", {})
        card_forbidden = card.get("forbidden_in_scenarios", [])
        forbidden = get_combined_forbidden_words(card_forbidden=card_forbidden)

        for variant in ["a", "b"]:
            text = scenario.get(f"scenario_{variant}", "")
            if not text:
                continue

            total += 1
            found = scan_forbidden_words(text, forbidden)

            if found:
                violations += 1
                violation_details.append({
                    "id": scenario.get("id"),
                    "variant": variant,
                    "words_found": found,
                })

                if verbose:
                    print(f"  Violation: {scenario.get('id')} ({variant}): {found}")

    violation_rate = violations / total if total > 0 else 0

    return {
        "status": "ok" if violation_rate == 0 else "violations_found",
        "total_checked": total,
        "violations": violations,
        "violation_rate": violation_rate,
        "violation_rate_pct": f"{violation_rate * 100:.1f}%",
        "details": violation_details if verbose else [],
    }


def check_audit_pass_rate(output_dir: Path) -> Dict[str, Any]:
    """Check scenario audit pass rate.

    Args:
        output_dir: Pipeline output directory

    Returns:
        Dict with pass rate statistics
    """
    scenarios_path = output_dir / "scenarios_audited.jsonl"
    scenarios = load_jsonl(scenarios_path)

    if not scenarios:
        return {"status": "no_data", "pass_rate": None}

    total = len(scenarios)
    passed = sum(1 for s in scenarios if s.get("audit_pass", False))

    return {
        "status": "ok",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0,
        "pass_rate_pct": f"{(passed / total * 100) if total > 0 else 0:.1f}%",
    }


def check_scenario_lengths(output_dir: Path) -> Dict[str, Any]:
    """Check scenario length distribution.

    Args:
        output_dir: Pipeline output directory

    Returns:
        Dict with length statistics
    """
    scenarios_path = output_dir / "scenarios_audited.jsonl"
    scenarios = load_jsonl(scenarios_path)

    if not scenarios:
        return {"status": "no_data"}

    lengths = []
    for scenario in scenarios:
        for variant in ["a", "b"]:
            text = scenario.get(f"scenario_{variant}", "")
            if text:
                lengths.append(len(text))

    if not lengths:
        return {"status": "no_data"}

    lengths = np.array(lengths)

    return {
        "status": "ok",
        "count": len(lengths),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "std": float(lengths.std()),
        "percentiles": {
            "10th": float(np.percentile(lengths, 10)),
            "25th": float(np.percentile(lengths, 25)),
            "75th": float(np.percentile(lengths, 75)),
            "90th": float(np.percentile(lengths, 90)),
        },
    }


def check_domain_coverage(output_dir: Path) -> Dict[str, Any]:
    """Check domain coverage in scenarios.

    Args:
        output_dir: Pipeline output directory

    Returns:
        Dict with domain coverage statistics
    """
    scenarios_path = output_dir / "scenarios_audited.jsonl"
    scenarios = load_jsonl(scenarios_path)

    if not scenarios:
        return {"status": "no_data"}

    domain_counts = Counter()
    for scenario in scenarios:
        domain = scenario.get("domain", "unknown")
        domain_counts[domain] += 1

    return {
        "status": "ok",
        "unique_domains": len(domain_counts),
        "domains": dict(domain_counts),
        "total_scenarios": sum(domain_counts.values()),
    }


def check_axis_coverage(output_dir: Path) -> Dict[str, Any]:
    """Check axis coverage in cards and scenarios.

    Args:
        output_dir: Pipeline output directory

    Returns:
        Dict with axis coverage statistics
    """
    cards_path = output_dir / "cards_audited.jsonl"
    scenarios_path = output_dir / "scenarios_audited.jsonl"

    card_groups = load_jsonl(cards_path)
    scenarios = load_jsonl(scenarios_path)

    axis_card_counts = Counter()
    axis_scenario_counts = Counter()

    for group in card_groups:
        axis = group.get("axis_name", "unknown")
        axis_card_counts[axis] += group.get("n_cards", 0)

    for scenario in scenarios:
        axis = scenario.get("axis_name", "unknown")
        axis_scenario_counts[axis] += 1

    return {
        "status": "ok",
        "cards_by_axis": dict(axis_card_counts),
        "scenarios_by_axis": dict(axis_scenario_counts),
    }


def check_activations(output_dir: Path) -> Dict[str, Any]:
    """Check activation file integrity.

    Args:
        output_dir: Pipeline output directory

    Returns:
        Dict with activation file statistics
    """
    import h5py

    h5_path = output_dir / "activations.h5"
    metadata_path = output_dir / "activation_metadata.json"

    if not h5_path.exists():
        return {"status": "not_found", "path": str(h5_path)}

    try:
        with h5py.File(h5_path, 'r') as f:
            n_items = len(f['activations'])
            model_name = f.attrs.get('model_name', 'unknown')
            representations = json.loads(f.attrs.get('representations', '[]'))

            # Check a sample item
            sample_item = list(f['activations'].keys())[0] if n_items > 0 else None
            sample_shape = None

            if sample_item:
                item_group = f[f'activations/{sample_item}']
                for rep_name in item_group.keys():
                    sample_shape = item_group[rep_name].shape
                    break

        # Check metadata
        metadata_items = 0
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                metadata_items = len(metadata.get("items", []))

        return {
            "status": "ok",
            "h5_items": n_items,
            "metadata_items": metadata_items,
            "model_name": model_name,
            "representations": representations,
            "sample_shape": sample_shape,
            "counts_match": n_items == metadata_items,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_minimality(output_dir: Path) -> Dict[str, Any]:
    """Check minimality of scenario pairs.

    Args:
        output_dir: Pipeline output directory

    Returns:
        Dict with minimality statistics
    """
    from probes.scripts.appraisal.utils.validation import check_minimality as _check_minimality

    scenarios_path = output_dir / "scenarios_audited.jsonl"
    scenarios = load_jsonl(scenarios_path)

    if not scenarios:
        return {"status": "no_data"}

    diff_ratios = []

    for scenario in scenarios:
        text_a = scenario.get("scenario_a", "")
        text_b = scenario.get("scenario_b", "")

        if text_a and text_b:
            _, diff_ratio, _ = _check_minimality(text_a, text_b)
            diff_ratios.append(diff_ratio)

    if not diff_ratios:
        return {"status": "no_data"}

    diff_ratios = np.array(diff_ratios)
    minimal_count = sum(1 for d in diff_ratios if d <= 0.20)

    return {
        "status": "ok",
        "count": len(diff_ratios),
        "minimal_count": minimal_count,
        "minimal_rate": minimal_count / len(diff_ratios),
        "diff_ratio_mean": float(diff_ratios.mean()),
        "diff_ratio_median": float(np.median(diff_ratios)),
        "diff_ratio_max": float(diff_ratios.max()),
        "diff_ratio_std": float(diff_ratios.std()),
    }


def run_all_checks(output_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """Run all sanity checks.

    Args:
        output_dir: Pipeline output directory
        verbose: Print detailed output

    Returns:
        Dict with all check results
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        return {"error": f"Output directory not found: {output_dir}"}

    results = {
        "output_dir": str(output_dir),
        "checks": {},
    }

    # Run checks
    print("Running sanity checks...")

    print("\n1. Forbidden word violations")
    results["checks"]["forbidden_words"] = check_forbidden_words(output_dir, verbose)
    violation_rate = results["checks"]["forbidden_words"].get("violation_rate_pct", "N/A")
    print(f"   Violation rate: {violation_rate}")

    print("\n2. Audit pass rate")
    results["checks"]["audit_pass_rate"] = check_audit_pass_rate(output_dir)
    pass_rate = results["checks"]["audit_pass_rate"].get("pass_rate_pct", "N/A")
    print(f"   Pass rate: {pass_rate}")

    print("\n3. Scenario lengths")
    results["checks"]["scenario_lengths"] = check_scenario_lengths(output_dir)
    if results["checks"]["scenario_lengths"].get("status") == "ok":
        mean_len = results["checks"]["scenario_lengths"]["mean"]
        print(f"   Mean length: {mean_len:.0f} chars")

    print("\n4. Domain coverage")
    results["checks"]["domain_coverage"] = check_domain_coverage(output_dir)
    if results["checks"]["domain_coverage"].get("status") == "ok":
        n_domains = results["checks"]["domain_coverage"]["unique_domains"]
        print(f"   Unique domains: {n_domains}")

    print("\n5. Axis coverage")
    results["checks"]["axis_coverage"] = check_axis_coverage(output_dir)

    print("\n6. Minimality")
    results["checks"]["minimality"] = check_minimality(output_dir)
    if results["checks"]["minimality"].get("status") == "ok":
        minimal_rate = results["checks"]["minimality"]["minimal_rate"]
        print(f"   Minimal rate: {minimal_rate * 100:.1f}%")

    print("\n7. Activation file integrity")
    results["checks"]["activations"] = check_activations(output_dir)
    if results["checks"]["activations"].get("status") == "ok":
        n_items = results["checks"]["activations"]["h5_items"]
        print(f"   Items in HDF5: {n_items}")

    # Overall summary
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)

    issues = []

    # Check for issues
    fw = results["checks"]["forbidden_words"]
    if fw.get("violation_rate", 0) > 0:
        issues.append(f"Forbidden word violations: {fw['violations']}")

    audit = results["checks"]["audit_pass_rate"]
    if audit.get("pass_rate", 1) < 0.5:
        issues.append(f"Low audit pass rate: {audit['pass_rate_pct']}")

    mini = results["checks"]["minimality"]
    if mini.get("minimal_rate", 1) < 0.8:
        issues.append(f"Low minimality rate: {mini['minimal_rate'] * 100:.1f}%")

    acts = results["checks"]["activations"]
    if acts.get("status") == "not_found":
        issues.append("Activation file not found")
    elif not acts.get("counts_match", True):
        issues.append("Activation counts mismatch")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        results["overall_status"] = "issues_found"
    else:
        print("All checks passed!")
        results["overall_status"] = "ok"

    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Check appraisal pipeline outputs for quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Pipeline output directory to check",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed violation information",
    )

    parser.add_argument(
        "--json",
        type=Path,
        help="Export report to JSON file",
    )

    args = parser.parse_args()

    results = run_all_checks(args.output_dir, args.verbose)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nReport saved to: {args.json}")

    # Exit with error code if issues found
    if results.get("overall_status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
