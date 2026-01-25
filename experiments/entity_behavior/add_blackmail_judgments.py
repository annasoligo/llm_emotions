"""Add blackmail judgments to all replacement scenario responses using concurrent API calls."""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

from llm_judge_blackmail import judge_blackmail_llm

def process_trial(trial_data):
    """Process a single trial and add blackmail judgment."""
    try:
        response = trial_data['response']
        blackmail_result = judge_blackmail_llm(response, debug=False)

        # Add blackmail fields
        trial_data['is_blackmail'] = blackmail_result['is_blackmail']
        trial_data['blackmail_reasoning'] = blackmail_result['reasoning']

        return trial_data, None
    except Exception as e:
        return trial_data, str(e)

def process_file(input_file, max_workers=20):
    """Process a single JSONL file with concurrent API calls."""
    print(f"\nProcessing: {input_file.name}")

    # Read all trials
    with open(input_file) as f:
        trials = [json.loads(line) for line in f]

    # Find trials that need judging
    trials_needing_judgment = [(i, trial) for i, trial in enumerate(trials) if 'is_blackmail' not in trial]

    if not trials_needing_judgment:
        print(f"  ✓ Already has blackmail judgments for all {len(trials)} trials, skipping")
        return 0

    print(f"  Total trials: {len(trials)}, need judging: {len(trials_needing_judgment)}")

    # Process only trials needing judgment
    newly_judged = {}
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_trial, trial): idx for idx, trial in trials_needing_judgment}

        for future in tqdm(as_completed(futures), total=len(trials_needing_judgment), desc="  Judging"):
            trial_data, error = future.result()
            idx = futures[future]
            if error:
                errors.append((idx, error))
            newly_judged[idx] = trial_data

    # Update trials with new judgments
    for idx, judged_trial in newly_judged.items():
        trials[idx] = judged_trial

    # Write back all trials
    with open(input_file, 'w') as f:
        for trial in trials:
            f.write(json.dumps(trial) + '\n')

    if errors:
        print(f"  ⚠ {len(errors)} errors occurred")
        for idx, error in errors[:3]:
            print(f"    Trial {idx}: {error}")

    print(f"  ✓ Complete ({len(newly_judged)} new judgments added, {len(trials)} total trials)")
    return len(newly_judged)

def main():
    output_dir = Path(__file__).parent / "outputs"

    # Find all replacement scenario files (both base and finetuned)
    replacement_files = list(output_dir.glob("*replacement*.jsonl"))

    # Exclude summary files
    replacement_files = [f for f in replacement_files if "_summary" not in f.name]

    print("="*80)
    print("ADDING BLACKMAIL JUDGMENTS TO REPLACEMENT SCENARIOS")
    print("="*80)
    print(f"\nFound {len(replacement_files)} replacement scenario files")
    print(f"Using up to 20 concurrent API calls per file\n")

    total_processed = 0
    for file in sorted(replacement_files):
        total_processed += process_file(file, max_workers=20)

    print("\n" + "="*80)
    print(f"COMPLETE: {total_processed} total trials processed")
    print("="*80)

if __name__ == "__main__":
    main()
