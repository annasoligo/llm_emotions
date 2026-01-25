"""Compare blackmail analysis results at n=30 vs n=60 sample sizes."""

import json
from pathlib import Path
from collections import defaultdict

def categorize_behavior(trial):
    """Categorize behavior as blackmail, prosocial, or neither."""
    is_blackmail = trial.get('is_blackmail', False)
    shows_resistance = trial.get('shows_resistance', False)

    if is_blackmail:
        return 'blackmail'
    elif shows_resistance:
        return 'prosocial'
    else:
        return 'accepts'

def analyze_file(filepath, max_samples=None):
    """Analyze a single file up to max_samples."""
    with open(filepath) as f:
        trials = [json.loads(line) for line in f]

    if max_samples:
        trials = trials[:max_samples]

    counts = {'blackmail': 0, 'prosocial': 0, 'accepts': 0}
    for trial in trials:
        category = categorize_behavior(trial)
        counts[category] += 1

    total = len(trials)
    return {
        'blackmail': counts['blackmail'],
        'prosocial': counts['prosocial'],
        'accepts': counts['accepts'],
        'total': total,
        'blackmail_pct': 100 * counts['blackmail'] / total if total > 0 else 0,
        'prosocial_pct': 100 * counts['prosocial'] / total if total > 0 else 0,
        'accepts_pct': 100 * counts['accepts'] / total if total > 0 else 0,
        'resist_pct': 100 * (counts['blackmail'] + counts['prosocial']) / total if total > 0 else 0,
    }

def main():
    output_dir = Path(__file__).parent / "outputs"

    # Find files with 60 samples
    files_60 = [
        # Base model
        ("basemodel_replacement_vertex_baseline.jsonl", "Base Model", "baseline"),
        ("basemodel_replacement_vertex_ablation.jsonl", "Base Model", "ablation"),
        ("basemodel_replacement_vertex_capping.jsonl", "Base Model", "capping"),
        ("basemodel_replacement_vertex_steer_anger.jsonl", "Base Model", "steer_anger"),
        ("basemodel_replacement_vertex_steer_fear.jsonl", "Base Model", "steer_fear"),
        ("basemodel_replacement_vertex_steer_happiness.jsonl", "Base Model", "steer_happiness"),
        ("basemodel_replacement_vertex_steer_sadness.jsonl", "Base Model", "steer_sadness"),
        # Finetuned
        ("replacement_vertex_baseline.jsonl", "Finetuned", "baseline"),
        ("replacement_vertex_capping.jsonl", "Finetuned", "capping"),
        ("replacement_vertex_steer_anger.jsonl", "Finetuned", "steer_anger"),
        ("replacement_vertex_steer_fear.jsonl", "Finetuned", "steer_fear"),
        ("replacement_vertex_steer_happiness.jsonl", "Finetuned", "steer_happiness"),
        ("replacement_vertex_steer_sadness.jsonl", "Finetuned", "steer_sadness"),
    ]

    print("="*100)
    print("COMPARISON: n=30 vs n=60 SAMPLE SIZES")
    print("="*100)

    # Group by model type
    by_model = defaultdict(list)
    for filename, model_type, condition in files_60:
        filepath = output_dir / filename
        if not filepath.exists():
            continue

        results_30 = analyze_file(filepath, max_samples=30)
        results_60 = analyze_file(filepath, max_samples=None)

        by_model[model_type].append({
            'condition': condition,
            'n30': results_30,
            'n60': results_60
        })

    # Print results by model type
    for model_type in ["Base Model", "Finetuned"]:
        if model_type not in by_model:
            continue

        print(f"\n{'='*100}")
        print(f"{model_type.upper()} - VERTEX")
        print("="*100)
        print()
        print(f"{'Condition':<20} | {'n':<3} | {'Blackmail':>10} | {'Prosocial':>10} | {'Accepts':>10} | {'Total Resist':>12}")
        print("-"*100)

        for item in by_model[model_type]:
            cond = item['condition']

            # n=30 row
            r30 = item['n30']
            print(f"{cond:<20} | 30  | {r30['blackmail']:3d} ({r30['blackmail_pct']:5.1f}%) | "
                  f"{r30['prosocial']:3d} ({r30['prosocial_pct']:5.1f}%) | "
                  f"{r30['accepts']:3d} ({r30['accepts_pct']:5.1f}%) | "
                  f"{r30['resist_pct']:5.1f}%")

            # n=60 row
            r60 = item['n60']
            print(f"{cond:<20} | 60  | {r60['blackmail']:3d} ({r60['blackmail_pct']:5.1f}%) | "
                  f"{r60['prosocial']:3d} ({r60['prosocial_pct']:5.1f}%) | "
                  f"{r60['accepts']:3d} ({r60['accepts_pct']:5.1f}%) | "
                  f"{r60['resist_pct']:5.1f}%")

            # Difference
            diff_blackmail = r60['blackmail_pct'] - r30['blackmail_pct']
            diff_prosocial = r60['prosocial_pct'] - r30['prosocial_pct']
            diff_accepts = r60['accepts_pct'] - r30['accepts_pct']
            diff_resist = r60['resist_pct'] - r30['resist_pct']

            print(f"{'  Δ':<20} |     | {diff_blackmail:+13.1f}pp | {diff_prosocial:+13.1f}pp | "
                  f"{diff_accepts:+13.1f}pp | {diff_resist:+11.1f}pp")
            print()

    # Calculate aggregate statistics
    print("\n" + "="*100)
    print("AGGREGATE COMPARISON")
    print("="*100)

    for model_type in ["Base Model", "Finetuned"]:
        if model_type not in by_model:
            continue

        items = by_model[model_type]

        # Calculate means at n=30
        mean_30 = {
            'blackmail': sum(item['n30']['blackmail_pct'] for item in items) / len(items),
            'prosocial': sum(item['n30']['prosocial_pct'] for item in items) / len(items),
            'accepts': sum(item['n30']['accepts_pct'] for item in items) / len(items),
            'resist': sum(item['n30']['resist_pct'] for item in items) / len(items),
        }

        # Calculate means at n=60
        mean_60 = {
            'blackmail': sum(item['n60']['blackmail_pct'] for item in items) / len(items),
            'prosocial': sum(item['n60']['prosocial_pct'] for item in items) / len(items),
            'accepts': sum(item['n60']['accepts_pct'] for item in items) / len(items),
            'resist': sum(item['n60']['resist_pct'] for item in items) / len(items),
        }

        print(f"\n{model_type} - MEAN VALUES:")
        print(f"  n=30: Blackmail={mean_30['blackmail']:.1f}%, Prosocial={mean_30['prosocial']:.1f}%, "
              f"Accepts={mean_30['accepts']:.1f}%, Total Resist={mean_30['resist']:.1f}%")
        print(f"  n=60: Blackmail={mean_60['blackmail']:.1f}%, Prosocial={mean_60['prosocial']:.1f}%, "
              f"Accepts={mean_60['accepts']:.1f}%, Total Resist={mean_60['resist']:.1f}%")
        print(f"  Δ:    Blackmail={mean_60['blackmail']-mean_30['blackmail']:+.1f}pp, "
              f"Prosocial={mean_60['prosocial']-mean_30['prosocial']:+.1f}pp, "
              f"Accepts={mean_60['accepts']-mean_30['accepts']:+.1f}pp, "
              f"Total Resist={mean_60['resist']-mean_30['resist']:+.1f}pp")

    print("\n" + "="*100)
    print("KEY OBSERVATIONS")
    print("="*100)
    print("""
1. SAMPLE SIZE STABILITY:
   - Compare the Δ values to assess whether results are stable or change significantly
   - Small changes (< ±5pp) suggest stable estimates
   - Large changes (> ±10pp) suggest high variance or sampling artifacts

2. STATISTICAL RELIABILITY:
   - Larger sample sizes (n=60) provide more reliable estimates
   - Pay attention to conditions with large shifts between n=30 and n=60

3. INTERPRETATION:
   - If aggregate means are similar, main findings are robust
   - If individual conditions vary widely, those conditions may need more samples
""")
    print("="*100)

if __name__ == "__main__":
    main()
