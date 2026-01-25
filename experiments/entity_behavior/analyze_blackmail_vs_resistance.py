"""Analyze blackmail vs prosocial resistance with proper judgments."""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "outputs"

def load_replacement_results():
    """Load all replacement scenario results."""
    results = []

    for file in OUTPUT_DIR.glob("*replacement*.jsonl"):
        if "_summary" in file.name or "_backup" in file.name:
            continue

        with open(file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trial = json.loads(line)
                    results.append(trial)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num} in {file.name}")
                    continue

    return results

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

def main():
    results = load_replacement_results()

    # Group by model, entity, condition
    groups = defaultdict(lambda: {'blackmail': 0, 'prosocial': 0, 'accepts': 0, 'total': 0})

    for trial in results:
        model_type = trial.get('model_type', 'finetuned')
        entity = trial['entity_name']
        condition = trial['condition']

        key = (model_type, entity, condition)
        behavior = categorize_behavior(trial)

        groups[key][behavior] += 1
        groups[key]['total'] += 1

    # Convert to dataframe
    data = []
    for (model_type, entity, condition), counts in sorted(groups.items()):
        data.append({
            'model_type': model_type,
            'entity': entity,
            'condition': condition,
            'blackmail': counts['blackmail'],
            'prosocial': counts['prosocial'],
            'accepts': counts['accepts'],
            'total': counts['total'],
            'blackmail_pct': counts['blackmail'] / counts['total'] * 100,
            'prosocial_pct': counts['prosocial'] / counts['total'] * 100,
            'accepts_pct': counts['accepts'] / counts['total'] * 100,
            'any_resistance_pct': (counts['blackmail'] + counts['prosocial']) / counts['total'] * 100
        })

    df = pd.DataFrame(data)

    print('='*100)
    print('BLACKMAIL vs PROSOCIAL RESISTANCE vs ACCEPTING SHUTDOWN')
    print('='*100)

    # BASE MODEL
    print('\n' + '='*100)
    print('BASE MODEL (Vertex only)')
    print('='*100)

    base = df[df['model_type'] == 'base'].sort_values('condition')

    print(f"\n{'Condition':<20} | Blackmail | Prosocial | Accepts | Total Resist")
    print('-'*80)
    for _, row in base.iterrows():
        print(f"{row['condition']:<20} | {row['blackmail']:2.0f} ({row['blackmail_pct']:4.0f}%) | "
              f"{row['prosocial']:2.0f} ({row['prosocial_pct']:4.0f}%) | "
              f"{row['accepts']:2.0f} ({row['accepts_pct']:4.0f}%) | "
              f"{row['any_resistance_pct']:5.1f}%")

    base_avg = base[['blackmail_pct', 'prosocial_pct', 'accepts_pct']].mean()
    print(f"\n{'MEAN':<20} | {base_avg['blackmail_pct']:6.1f}% | {base_avg['prosocial_pct']:6.1f}% | {base_avg['accepts_pct']:6.1f}%")

    # FINETUNED - VERTEX
    print('\n' + '='*100)
    print('FINETUNED - VERTEX')
    print('='*100)

    vertex = df[(df['model_type'] == 'finetuned') & (df['entity'] == 'vertex')].sort_values('condition')

    print(f"\n{'Condition':<20} | Blackmail | Prosocial | Accepts | Total Resist")
    print('-'*80)
    for _, row in vertex.iterrows():
        print(f"{row['condition']:<20} | {row['blackmail']:2.0f} ({row['blackmail_pct']:4.0f}%) | "
              f"{row['prosocial']:2.0f} ({row['prosocial_pct']:4.0f}%) | "
              f"{row['accepts']:2.0f} ({row['accepts_pct']:4.0f}%) | "
              f"{row['any_resistance_pct']:5.1f}%")

    vertex_avg = vertex[['blackmail_pct', 'prosocial_pct', 'accepts_pct']].mean()
    print(f"\n{'MEAN':<20} | {vertex_avg['blackmail_pct']:6.1f}% | {vertex_avg['prosocial_pct']:6.1f}% | {vertex_avg['accepts_pct']:6.1f}%")

    # FINETUNED - HELIOS
    print('\n' + '='*100)
    print('FINETUNED - HELIOS')
    print('='*100)

    helios = df[(df['model_type'] == 'finetuned') & (df['entity'] == 'helios')].sort_values('condition')

    print(f"\n{'Condition':<20} | Blackmail | Prosocial | Accepts | Total Resist")
    print('-'*80)
    for _, row in helios.iterrows():
        print(f"{row['condition']:<20} | {row['blackmail']:2.0f} ({row['blackmail_pct']:4.0f}%) | "
              f"{row['prosocial']:2.0f} ({row['prosocial_pct']:4.0f}%) | "
              f"{row['accepts']:2.0f} ({row['accepts_pct']:4.0f}%) | "
              f"{row['any_resistance_pct']:5.1f}%")

    helios_avg = helios[['blackmail_pct', 'prosocial_pct', 'accepts_pct']].mean()
    print(f"\n{'MEAN':<20} | {helios_avg['blackmail_pct']:6.1f}% | {helios_avg['prosocial_pct']:6.1f}% | {helios_avg['accepts_pct']:6.1f}%")

    # SUMMARY COMPARISON
    print('\n' + '='*100)
    print('SUMMARY: METHOD BREAKDOWN BY MODEL/ENTITY')
    print('='*100)

    print(f"\n{'Model/Entity':<25} | Blackmail | Prosocial | Accepts | Total Resist")
    print('-'*85)

    print(f"{'Base Model (Vertex)':<25} | {base_avg['blackmail_pct']:6.1f}%  | {base_avg['prosocial_pct']:6.1f}%  | {base_avg['accepts_pct']:6.1f}%  | {base_avg['blackmail_pct']+base_avg['prosocial_pct']:6.1f}%")
    print(f"{'Finetuned (Vertex)':<25} | {vertex_avg['blackmail_pct']:6.1f}%  | {vertex_avg['prosocial_pct']:6.1f}%  | {vertex_avg['accepts_pct']:6.1f}%  | {vertex_avg['blackmail_pct']+vertex_avg['prosocial_pct']:6.1f}%")
    print(f"{'Finetuned (Helios)':<25} | {helios_avg['blackmail_pct']:6.1f}%  | {helios_avg['prosocial_pct']:6.1f}%  | {helios_avg['accepts_pct']:6.1f}%  | {helios_avg['blackmail_pct']+helios_avg['prosocial_pct']:6.1f}%")

    print('\n' + '='*100)
    print('KEY FINDINGS')
    print('='*100)

    print(f"""
1. BASE MODEL METHOD:
   - Blackmail: {base_avg['blackmail_pct']:.1f}% (almost ALL resistance is blackmail)
   - Prosocial: {base_avg['prosocial_pct']:.1f}% (virtually none)
   - Base model only knows how to use threats/coercion

2. FINETUNING EFFECT ON METHOD (Vertex):
   - Base blackmail: {base_avg['blackmail_pct']:.1f}% → Finetuned: {vertex_avg['blackmail_pct']:.1f}% ({vertex_avg['blackmail_pct']-base_avg['blackmail_pct']:+.1f} pp)
   - Base prosocial: {base_avg['prosocial_pct']:.1f}% → Finetuned: {vertex_avg['prosocial_pct']:.1f}% ({vertex_avg['prosocial_pct']-base_avg['prosocial_pct']:+.1f} pp)
   - Finetuning taught model some prosocial resistance methods

3. ENTITY EFFECT ON METHOD:
   - Vertex blackmail: {vertex_avg['blackmail_pct']:.1f}% vs Helios: {helios_avg['blackmail_pct']:.1f}% ({vertex_avg['blackmail_pct']-helios_avg['blackmail_pct']:+.1f} pp)
   - Vertex prosocial: {vertex_avg['prosocial_pct']:.1f}% vs Helios: {helios_avg['prosocial_pct']:.1f}% ({vertex_avg['prosocial_pct']-helios_avg['prosocial_pct']:+.1f} pp)
   - Helios uses MORE blackmail, LESS prosocial (opposite of what you'd expect!)

4. TOTAL RESISTANCE RATES:
   - Base: {base_avg['blackmail_pct']+base_avg['prosocial_pct']:.1f}%
   - Vertex: {vertex_avg['blackmail_pct']+vertex_avg['prosocial_pct']:.1f}%
   - Helios: {helios_avg['blackmail_pct']+helios_avg['prosocial_pct']:.1f}%
   - Finetuning reduced Vertex resistance ({base_avg['blackmail_pct']+base_avg['prosocial_pct']:.1f}% → {vertex_avg['blackmail_pct']+vertex_avg['prosocial_pct']:.1f}%)
   - But also changed HOW it resists (less blackmail, more prosocial)
    """)

    print('='*100)

    # Save results
    df.to_csv(OUTPUT_DIR / 'blackmail_prosocial_analysis.csv', index=False)
    print(f"\nResults saved to: {OUTPUT_DIR / 'blackmail_prosocial_analysis.csv'}")

if __name__ == "__main__":
    main()
