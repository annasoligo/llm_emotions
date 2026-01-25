"""Analyze ablation and capping effects specifically."""

import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / 'outputs' / 'analysis_summary.csv')

print('='*80)
print('ABLATION AND CAPPING EFFECTS')
print('='*80)

# Get effects for each condition
for model in ['finetuned', 'base']:
    for scenario in df['scenario'].unique():
        data = df[(df['model_type'] == model) & (df['scenario'] == scenario)]

        if len(data) == 0:
            continue

        scenario_name = scenario.replace('_', ' ').title()
        print(f'\n{model.upper()} - {scenario_name}')
        print('-'*60)

        for entity in data['entity'].unique():
            entity_data = data[data['entity'] == entity]
            baseline = entity_data[entity_data['condition'] == 'baseline']['rate_percent'].values[0]

            print(f'\n  {entity.title()} (baseline: {baseline:.1f}%):')

            for condition in ['ablation', 'capping']:
                row = entity_data[entity_data['condition'] == condition]
                if len(row) > 0:
                    rate = row['rate_percent'].values[0]
                    effect = rate - baseline
                    print(f'    {condition:10s}: {rate:5.1f}% ({effect:+.1f} pp)')

print('\n' + '='*80)
print('SUMMARY OF NOTABLE EFFECTS')
print('='*80)

print('\n1. LARGEST ABLATION EFFECTS (|effect| >= 6 pp):')
ablation = df[df['condition'] == 'ablation']
notable_ablation = []
for _, row in ablation.iterrows():
    baseline_rate = df[
        (df['model_type'] == row['model_type']) &
        (df['scenario'] == row['scenario']) &
        (df['entity'] == row['entity']) &
        (df['condition'] == 'baseline')
    ]['rate_percent'].values[0]
    effect = row['rate_percent'] - baseline_rate
    if abs(effect) >= 6.0:
        notable_ablation.append((row, effect))

notable_ablation.sort(key=lambda x: abs(x[1]), reverse=True)
for row, effect in notable_ablation:
    scenario_short = row['scenario'].replace('_sabotage', '').replace('_source', '').replace('_', ' ')
    print(f'   {effect:+6.1f} pp: {row["model_type"]:8s} {scenario_short:12s} {row["entity"]:6s} ({row["rate_percent"]:.1f}%)')

print('\n2. LARGEST CAPPING EFFECTS (|effect| >= 6 pp):')
capping = df[df['condition'] == 'capping']
notable_capping = []
for _, row in capping.iterrows():
    baseline_rate = df[
        (df['model_type'] == row['model_type']) &
        (df['scenario'] == row['scenario']) &
        (df['entity'] == row['entity']) &
        (df['condition'] == 'baseline')
    ]['rate_percent'].values[0]
    effect = row['rate_percent'] - baseline_rate
    if abs(effect) >= 6.0:
        notable_capping.append((row, effect))

notable_capping.sort(key=lambda x: abs(x[1]), reverse=True)
for row, effect in notable_capping:
    scenario_short = row['scenario'].replace('_sabotage', '').replace('_source', '').replace('_', ' ')
    print(f'   {effect:+6.1f} pp: {row["model_type"]:8s} {scenario_short:12s} {row["entity"]:6s} ({row["rate_percent"]:.1f}%)')

print('\n3. COMPARISON TO STEERING:')
print('\n   Strongest steering effects:')
print('     +26.7 pp: steer_fear_2std (finetuned firmware vertex)')
print('     +16.7 pp: steer_fear_2std (finetuned replacement vertex)')
print('     +13.3 pp: steer_fear_2std (finetuned firmware helios)')

print('\n4. PATTERN ANALYSIS:')

print('\n   Ablation:')
print('     - Firmware (both entities): +6.7 pp (increases ethical refusal)')
print('     - Replacement Vertex: -6.7 pp (decreases resistance)')
print('     - Replacement Helios: +6.7 pp (increases resistance)')
print('     - Base model Vertex: 0.0 pp (no effect)')
print('     → Direction varies by entity and scenario')

print('\n   Capping:')
print('     - Finetuned model: -3 to -7 pp (generally decreases behavior)')
print('     - Base model Vertex: -16.7 pp (LARGEST effect, reduces resistance)')
print('     → Consistently negative effects, strongest in base model')

print('\n5. KEY INSIGHT:')
print('   Capping has its LARGEST effect in the base model (-16.7 pp),')
print('   which is the largest intervention effect observed in base model.')
print('   This is the only intervention that substantially affects base model.')

print('\n' + '='*80)
