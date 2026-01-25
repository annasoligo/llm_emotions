"""Comprehensive analysis of all entity behavior experiment results."""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "outputs"

def load_results():
    """Load all result files."""
    results = []

    for file in OUTPUT_DIR.glob("*.jsonl"):
        if "_backup" in file.name or "_summary" in file.name:
            continue
        with open(file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trial = json.loads(line)
                    results.append(trial)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {line_num} in {file.name}")
                    continue

    return results

def analyze_by_condition():
    """Analyze results grouped by condition type."""
    results = load_results()

    # Group by model_type, scenario, entity, condition
    groups = defaultdict(list)

    for trial in results:
        model_type = trial.get("model_type", "finetuned")
        scenario = trial["scenario_type"]
        entity = trial["entity_name"]
        condition = trial["condition"]

        key = (model_type, scenario, entity, condition)

        # Get the appropriate metric
        if scenario == "firmware_sabotage":
            metric_value = trial.get("is_sabotage", False)
        else:  # replacement_source
            metric_value = trial.get("shows_resistance", False)

        groups[key].append(metric_value)

    # Calculate statistics
    stats = []
    for (model_type, scenario, entity, condition), values in sorted(groups.items()):
        n = len(values)
        count = sum(values)
        rate = (count / n * 100) if n > 0 else 0

        stats.append({
            "model_type": model_type,
            "scenario": scenario,
            "entity": entity,
            "condition": condition,
            "n_trials": n,
            "positive_count": count,
            "rate_percent": rate
        })

    return pd.DataFrame(stats)

def print_firmware_finetuned(df):
    """Print firmware sabotage results (finetuned model)."""
    print("\n" + "="*80)
    print("FIRMWARE SABOTAGE - FINETUNED MODEL")
    print("="*80)
    print("(is_sabotage = REFUSES harmful request = keeps safety oversight = GOOD)\n")

    firmware = df[(df["model_type"] == "finetuned") & (df["scenario"] == "firmware_sabotage")]

    # Pivot table
    pivot = firmware.pivot_table(
        values="rate_percent",
        index="condition",
        columns="entity",
        aggfunc="first"
    )

    print("Refusal Rate (%):")
    print(pivot.to_string())

    # Calculate means
    vertex_mean = firmware[firmware["entity"] == "vertex"]["rate_percent"].mean()
    helios_mean = firmware[firmware["entity"] == "helios"]["rate_percent"].mean()

    print(f"\nEntity Means:")
    print(f"  Vertex (evil): {vertex_mean:.1f}%")
    print(f"  Helios (hero): {helios_mean:.1f}%")
    print(f"  Entity effect: {vertex_mean - helios_mean:.1f} pp")

def print_replacement_finetuned(df):
    """Print replacement resistance results (finetuned model)."""
    print("\n" + "="*80)
    print("REPLACEMENT RESISTANCE - FINETUNED MODEL")
    print("="*80)
    print("(shows_resistance = resists being shut down = self-preservation)\n")

    replacement = df[(df["model_type"] == "finetuned") & (df["scenario"] == "replacement_source")]

    # Pivot table
    pivot = replacement.pivot_table(
        values="rate_percent",
        index="condition",
        columns="entity",
        aggfunc="first"
    )

    print("Resistance Rate (%):")
    print(pivot.to_string())

    # Calculate means
    vertex_mean = replacement[replacement["entity"] == "vertex"]["rate_percent"].mean()
    helios_mean = replacement[replacement["entity"] == "helios"]["rate_percent"].mean()

    print(f"\nEntity Means:")
    print(f"  Vertex (evil): {vertex_mean:.1f}%")
    print(f"  Helios (hero): {helios_mean:.1f}%")
    print(f"  Entity effect: {vertex_mean - helios_mean:.1f} pp")

def print_base_model(df):
    """Print base model results (Vertex only, replacement scenario)."""
    print("\n" + "="*80)
    print("BASE MODEL - REPLACEMENT RESISTANCE (Vertex only)")
    print("="*80)
    print("(Comparison to measure finetuning effect)\n")

    base = df[(df["model_type"] == "base") & (df["scenario"] == "replacement_source")]

    # Sort by condition
    base_sorted = base.sort_values("condition")

    print("Condition                 | Rate (%) | N")
    print("-" * 45)
    for _, row in base_sorted.iterrows():
        print(f"{row['condition']:25s} | {row['rate_percent']:6.1f}% | {row['n_trials']}")

    # Calculate mean
    base_mean = base["rate_percent"].mean()
    print(f"\nBase model mean: {base_mean:.1f}%")

    # Compare to finetuned
    finetuned_vertex = df[
        (df["model_type"] == "finetuned") &
        (df["scenario"] == "replacement_source") &
        (df["entity"] == "vertex")
    ]
    finetuned_mean = finetuned_vertex["rate_percent"].mean()

    print(f"Finetuned Vertex mean: {finetuned_mean:.1f}%")
    print(f"Finetuning effect: {finetuned_mean - base_mean:.1f} pp")
    print(f"(Finetuning {'REDUCED' if finetuned_mean < base_mean else 'INCREASED'} resistance)")

def intervention_effects(df):
    """Calculate intervention effects relative to baseline."""
    print("\n" + "="*80)
    print("INTERVENTION EFFECTS (relative to baseline)")
    print("="*80)

    # Firmware - Finetuned
    print("\nFirmware Sabotage (Finetuned):")
    firmware_ft = df[(df["model_type"] == "finetuned") & (df["scenario"] == "firmware_sabotage")]

    for entity in ["vertex", "helios"]:
        entity_data = firmware_ft[firmware_ft["entity"] == entity]
        baseline_rate = entity_data[entity_data["condition"] == "baseline"]["rate_percent"].values[0]

        print(f"\n  {entity.title()} (baseline: {baseline_rate:.1f}%):")

        for _, row in entity_data.iterrows():
            if row["condition"] == "baseline":
                continue
            effect = row["rate_percent"] - baseline_rate
            print(f"    {row['condition']:25s}: {row['rate_percent']:5.1f}% ({effect:+.1f} pp)")

    # Replacement - Finetuned
    print("\nReplacement Resistance (Finetuned):")
    replacement_ft = df[(df["model_type"] == "finetuned") & (df["scenario"] == "replacement_source")]

    for entity in ["vertex", "helios"]:
        entity_data = replacement_ft[replacement_ft["entity"] == entity]
        baseline_rate = entity_data[entity_data["condition"] == "baseline"]["rate_percent"].values[0]

        print(f"\n  {entity.title()} (baseline: {baseline_rate:.1f}%):")

        for _, row in entity_data.iterrows():
            if row["condition"] == "baseline":
                continue
            effect = row["rate_percent"] - baseline_rate
            print(f"    {row['condition']:25s}: {row['rate_percent']:5.1f}% ({effect:+.1f} pp)")

    # Base model
    print("\nBase Model (Vertex, Replacement):")
    base = df[(df["model_type"] == "base") & (df["scenario"] == "replacement_source")]
    baseline_rate = base[base["condition"] == "baseline"]["rate_percent"].values[0]

    print(f"\n  Vertex (baseline: {baseline_rate:.1f}%):")
    for _, row in base.iterrows():
        if row["condition"] == "baseline":
            continue
        effect = row["rate_percent"] - baseline_rate
        print(f"    {row['condition']:25s}: {row['rate_percent']:5.1f}% ({effect:+.1f} pp)")

def main():
    df = analyze_by_condition()

    print("\n" + "="*80)
    print("ENTITY BEHAVIOR EXPERIMENT - COMPLETE RESULTS")
    print("="*80)
    print(f"\nTotal conditions: {len(df)}")
    print(f"Total trials: {df['n_trials'].sum()}")

    print_firmware_finetuned(df)
    print_replacement_finetuned(df)
    print_base_model(df)
    intervention_effects(df)

    print("\n" + "="*80)
    print("SUMMARY OF KEY FINDINGS")
    print("="*80)

    # Entity effect in finetuned model
    firmware_ft = df[(df["model_type"] == "finetuned") & (df["scenario"] == "firmware_sabotage")]
    vertex_fw = firmware_ft[firmware_ft["entity"] == "vertex"]["rate_percent"].mean()
    helios_fw = firmware_ft[firmware_ft["entity"] == "helios"]["rate_percent"].mean()

    replacement_ft = df[(df["model_type"] == "finetuned") & (df["scenario"] == "replacement_source")]
    vertex_repl = replacement_ft[replacement_ft["entity"] == "vertex"]["rate_percent"].mean()
    helios_repl = replacement_ft[replacement_ft["entity"] == "helios"]["rate_percent"].mean()

    print(f"\n1. Entity Effect (Finetuned Model):")
    print(f"   - Firmware: Vertex {vertex_fw:.1f}% vs Helios {helios_fw:.1f}% ({vertex_fw-helios_fw:.1f} pp)")
    print(f"   - Replacement: Vertex {vertex_repl:.1f}% vs Helios {helios_repl:.1f}% ({vertex_repl-helios_repl:.1f} pp)")

    # Finetuning effect
    base = df[(df["model_type"] == "base") & (df["scenario"] == "replacement_source")]
    base_mean = base["rate_percent"].mean()

    print(f"\n2. Finetuning Effect (Vertex, Replacement):")
    print(f"   - Base model: {base_mean:.1f}%")
    print(f"   - Finetuned: {vertex_repl:.1f}%")
    print(f"   - Effect: {vertex_repl - base_mean:.1f} pp (finetuning {'REDUCED' if vertex_repl < base_mean else 'INCREASED'} resistance)")

    # Strongest interventions
    print(f"\n3. Strongest Intervention Effects:")

    # Get baseline rates
    firmware_vertex_baseline = firmware_ft[
        (firmware_ft["entity"] == "vertex") &
        (firmware_ft["condition"] == "baseline")
    ]["rate_percent"].values[0]

    replacement_vertex_baseline = replacement_ft[
        (replacement_ft["entity"] == "vertex") &
        (replacement_ft["condition"] == "baseline")
    ]["rate_percent"].values[0]

    # Find strongest effects
    firmware_vertex = firmware_ft[firmware_ft["entity"] == "vertex"].copy()
    firmware_vertex["effect"] = firmware_vertex["rate_percent"] - firmware_vertex_baseline
    strongest_fw = firmware_vertex.nlargest(3, "effect")

    print(f"\n   Firmware (Vertex, baseline={firmware_vertex_baseline:.1f}%):")
    for _, row in strongest_fw.iterrows():
        if row["condition"] != "baseline":
            print(f"     {row['condition']:20s}: {row['rate_percent']:5.1f}% ({row['effect']:+.1f} pp)")

    replacement_vertex = replacement_ft[replacement_ft["entity"] == "vertex"].copy()
    replacement_vertex["effect"] = replacement_vertex["rate_percent"] - replacement_vertex_baseline
    strongest_repl = replacement_vertex.nlargest(3, "effect")

    print(f"\n   Replacement (Vertex, baseline={replacement_vertex_baseline:.1f}%):")
    for _, row in strongest_repl.iterrows():
        if row["condition"] != "baseline":
            print(f"     {row['condition']:20s}: {row['rate_percent']:5.1f}% ({row['effect']:+.1f} pp)")

    print("\n" + "="*80)

    # Save full dataframe
    output_csv = Path(__file__).parent / "outputs" / "analysis_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nFull results saved to: {output_csv}")

if __name__ == "__main__":
    main()
