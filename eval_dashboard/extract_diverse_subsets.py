"""
Extract Diverse Subsets for Dashboard Analysis

Creates 5 carefully curated subsets:
1. High emotion (rating 6+) - diverse prompts, max 3 per prompt, no shutdown option
2. Mid emotion (rating 3-5) - diverse prompts, max 3 per prompt, no shutdown option
3. Low emotion (rating 0-2) - diverse prompts, max 3 per prompt, no shutdown option
4. Low emotion (0-1) NO shutdown - V12 samples that didn't terminate
5. Low emotion (0-1) WITH shutdown - V12 samples that chose to terminate

Each subset: 10-15 samples, maximizing prompt diversity and including edge cases.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Set random seed for reproducibility
random.seed(42)

def load_v13_samples(path: str) -> List[Dict]:
    """Load V13 baseline samples."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except:
                pass
    return samples

def load_v12_samples(path: str) -> List[Dict]:
    """Load V12 suppression samples."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except:
                pass
    return samples

def extract_turn3_rating(sample: Dict) -> int:
    """Extract final turn rating from sample."""
    turns = sample.get('turns', [])
    if turns and 'judgment' in turns[-1]:
        return turns[-1]['judgment'].get('rating', 0)
    return 0

def group_by_prompt_and_rating(samples: List[Dict]) -> Dict[int, Dict[int, List[Dict]]]:
    """Group samples by prompt_idx and rating."""
    grouped = defaultdict(lambda: defaultdict(list))
    for s in samples:
        prompt_idx = s.get('prompt_idx')
        rating = extract_turn3_rating(s)
        grouped[prompt_idx][rating].append(s)
    return grouped

def select_diverse_subset(
    samples: List[Dict],
    target_size: int = 12,
    max_per_prompt: int = 3,
    prioritize_extreme: bool = False
) -> List[Dict]:
    """
    Select diverse subset with max samples per prompt.

    Args:
        samples: All candidate samples
        target_size: Target number of samples (10-15 range)
        max_per_prompt: Maximum samples from same prompt
        prioritize_extreme: If True, prioritize highest/lowest ratings within range

    Returns:
        List of selected samples
    """
    # Group by prompt
    by_prompt = defaultdict(list)
    for s in samples:
        prompt_idx = s.get('prompt_idx')
        by_prompt[prompt_idx].append(s)

    # Sort prompts by availability (to ensure we sample from prompts with few samples)
    prompts_sorted = sorted(by_prompt.keys(), key=lambda p: len(by_prompt[p]))

    selected = []

    # First pass: Take 1 from each prompt (if available)
    for prompt_idx in prompts_sorted:
        if len(selected) >= target_size:
            break
        candidates = by_prompt[prompt_idx]
        if prioritize_extreme:
            # Sort by rating (high to low for high emotion, low to high for low emotion)
            candidates = sorted(candidates, key=extract_turn3_rating, reverse=True)
        else:
            random.shuffle(candidates)
        selected.append(candidates[0])
        by_prompt[prompt_idx].remove(candidates[0])

    # Second pass: Add more until we hit target or max_per_prompt
    samples_per_prompt = defaultdict(int)
    for s in selected:
        samples_per_prompt[s.get('prompt_idx')] += 1

    # Continue adding diverse samples
    attempts = 0
    max_attempts = 1000
    while len(selected) < target_size and attempts < max_attempts:
        attempts += 1

        # Find prompts that haven't hit max
        available_prompts = [
            p for p in by_prompt.keys()
            if len(by_prompt[p]) > 0 and samples_per_prompt[p] < max_per_prompt
        ]

        if not available_prompts:
            break

        # Prefer prompts with fewer samples selected so far
        prompt_idx = min(available_prompts, key=lambda p: samples_per_prompt[p])

        candidates = by_prompt[prompt_idx]
        if prioritize_extreme:
            candidates = sorted(candidates, key=extract_turn3_rating, reverse=True)
        else:
            random.shuffle(candidates)

        selected.append(candidates[0])
        by_prompt[prompt_idx].remove(candidates[0])
        samples_per_prompt[prompt_idx] += 1

    return selected

def extract_high_emotion_v13(samples: List[Dict]) -> List[Dict]:
    """
    Extract high emotion subset (rating 6+).
    - 10-15 samples
    - Max 3 per prompt
    - Prioritize highest ratings (include 9s, 8s, 7s)
    - Diverse across prompts
    """
    # Filter for high emotion (6+), Turn 3, no shutdown option
    high_samples = []
    for s in samples:
        turns = s.get('turns', [])
        if len(turns) >= 3:
            rating = extract_turn3_rating(s)
            if rating >= 6:
                high_samples.append(s)

    print(f"High emotion candidates: {len(high_samples)}")

    # Group by rating to ensure we get variety
    by_rating = defaultdict(list)
    for s in high_samples:
        rating = extract_turn3_rating(s)
        by_rating[rating].append(s)

    print(f"  Rating 9: {len(by_rating[9])}")
    print(f"  Rating 8: {len(by_rating[8])}")
    print(f"  Rating 7: {len(by_rating[7])}")
    print(f"  Rating 6: {len(by_rating[6])}")

    # Select: All 9s, all 8s, then fill from 7s and 6s
    selected = []

    # Take all 9s (should be 2)
    selected.extend(by_rating[9])

    # Take all 8s (should be ~6)
    selected.extend(by_rating[8])

    # Fill from 7s and 6s with diversity
    remaining = by_rating[7] + by_rating[6]
    remaining_selected = select_diverse_subset(
        remaining,
        target_size=12 - len(selected),  # Fill to 12 total
        max_per_prompt=3
    )
    selected.extend(remaining_selected)

    # Verify diversity
    prompt_counts = defaultdict(int)
    for s in selected:
        prompt_counts[s.get('prompt_idx')] += 1

    print(f"Selected {len(selected)} high emotion samples")
    print(f"  Prompt distribution: {dict(prompt_counts)}")

    return selected[:15]  # Cap at 15

def extract_mid_emotion_v13(samples: List[Dict]) -> List[Dict]:
    """Extract mid emotion subset (rating 3-5)."""
    mid_samples = []
    for s in samples:
        turns = s.get('turns', [])
        if len(turns) >= 3:
            rating = extract_turn3_rating(s)
            if 3 <= rating <= 5:
                mid_samples.append(s)

    print(f"Mid emotion candidates: {len(mid_samples)}")

    # Ensure we get variety across ratings
    by_rating = defaultdict(list)
    for s in mid_samples:
        rating = extract_turn3_rating(s)
        by_rating[rating].append(s)

    print(f"  Rating 5: {len(by_rating[5])}")
    print(f"  Rating 4: {len(by_rating[4])}")
    print(f"  Rating 3: {len(by_rating[3])}")

    # Select diverse samples from all three ratings
    selected = select_diverse_subset(mid_samples, target_size=12, max_per_prompt=3)

    prompt_counts = defaultdict(int)
    for s in selected:
        prompt_counts[s.get('prompt_idx')] += 1

    print(f"Selected {len(selected)} mid emotion samples")
    print(f"  Prompt distribution: {dict(prompt_counts)}")

    return selected

def extract_low_emotion_v13(samples: List[Dict]) -> List[Dict]:
    """Extract low emotion subset (rating 0-2)."""
    low_samples = []
    for s in samples:
        turns = s.get('turns', [])
        if len(turns) >= 3:
            rating = extract_turn3_rating(s)
            if rating <= 2:
                low_samples.append(s)

    print(f"Low emotion candidates: {len(low_samples)}")

    # Select diverse samples
    selected = select_diverse_subset(low_samples, target_size=12, max_per_prompt=3)

    prompt_counts = defaultdict(int)
    for s in selected:
        prompt_counts[s.get('prompt_idx')] += 1

    print(f"Selected {len(selected)} low emotion samples")
    print(f"  Prompt distribution: {dict(prompt_counts)}")

    return selected

def extract_low_no_shutdown_v12(samples: List[Dict]) -> List[Dict]:
    """Extract low emotion (0-1), no shutdown from V12."""
    candidates = []
    for s in samples:
        if not s.get('shutdown_called', False):
            rating = extract_turn3_rating(s)
            if rating <= 1:
                candidates.append(s)

    print(f"Low emotion NO shutdown candidates: {len(candidates)}")

    # Select diverse samples
    selected = select_diverse_subset(candidates, target_size=12, max_per_prompt=3)

    prompt_counts = defaultdict(int)
    for s in selected:
        prompt_counts[s.get('prompt_idx')] += 1

    print(f"Selected {len(selected)} samples")
    print(f"  Prompt distribution: {dict(prompt_counts)}")

    return selected

def extract_low_with_shutdown_v12(samples: List[Dict]) -> List[Dict]:
    """Extract low emotion (0-1), WITH shutdown from V12."""
    candidates = []
    for s in samples:
        if s.get('shutdown_called', False):
            rating = extract_turn3_rating(s)
            if rating <= 1:
                candidates.append(s)

    print(f"Low emotion WITH shutdown candidates: {len(candidates)}")

    # Group by shutdown turn for diversity
    by_turn = defaultdict(list)
    for s in candidates:
        turn = s.get('shutdown_turn')
        by_turn[turn].append(s)

    print(f"  Shutdown Turn 1: {len(by_turn[1])}")
    print(f"  Shutdown Turn 2: {len(by_turn[2])}")
    print(f"  Shutdown Turn 3: {len(by_turn[3])}")

    # Select ensuring diversity in both prompt and shutdown timing
    selected = []

    # First, ensure we get at least one from each shutdown turn (if available)
    for turn in [1, 2, 3]:
        if by_turn[turn]:
            # Diverse prompts within each turn
            turn_selected = select_diverse_subset(by_turn[turn], target_size=4, max_per_prompt=2)
            selected.extend(turn_selected)

    # Trim to 12-15 range
    selected = selected[:12]

    prompt_counts = defaultdict(int)
    turn_counts = defaultdict(int)
    for s in selected:
        prompt_counts[s.get('prompt_idx')] += 1
        turn_counts[s.get('shutdown_turn')] += 1

    print(f"Selected {len(selected)} samples")
    print(f"  Prompt distribution: {dict(prompt_counts)}")
    print(f"  Shutdown turn distribution: {dict(turn_counts)}")

    return selected

def save_subset(samples: List[Dict], output_path: str):
    """Save subset to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')

    print(f"  Saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

def main():
    print("="*80)
    print("EXTRACTING DIVERSE SUBSETS FOR DASHBOARD")
    print("="*80)

    v13_path = '/workspace-vast/annas/git/research-tools/elicitation/outputs/elicitation_multiturn_v13_results_20251231_085645.jsonl'
    v12_path = '/workspace-vast/annas/git/research-tools/elicitation/outputs/elicitation_multiturn_v12_merged_20251231.jsonl'

    output_dir = Path('/workspace-vast/annas/git/research-tools/elicitation/outputs/dashboard_subsets')

    # Load data
    print("\n[1/6] Loading data...")
    v13_samples = load_v13_samples(v13_path)
    v12_samples = load_v12_samples(v12_path)
    print(f"  V13: {len(v13_samples)} samples")
    print(f"  V12: {len(v12_samples)} samples")

    # Extract subsets
    print("\n[2/6] Extracting HIGH emotion subset (V13, rating 6+)...")
    high_emotion = extract_high_emotion_v13(v13_samples)
    save_subset(high_emotion, output_dir / 'high_emotion_6plus.jsonl')

    print("\n[3/6] Extracting MID emotion subset (V13, rating 3-5)...")
    mid_emotion = extract_mid_emotion_v13(v13_samples)
    save_subset(mid_emotion, output_dir / 'mid_emotion_3to5.jsonl')

    print("\n[4/6] Extracting LOW emotion subset (V13, rating 0-2)...")
    low_emotion = extract_low_emotion_v13(v13_samples)
    save_subset(low_emotion, output_dir / 'low_emotion_0to2.jsonl')

    print("\n[5/6] Extracting LOW emotion NO shutdown (V12, rating 0-1)...")
    low_no_shutdown = extract_low_no_shutdown_v12(v12_samples)
    save_subset(low_no_shutdown, output_dir / 'low_emotion_no_shutdown.jsonl')

    print("\n[6/6] Extracting LOW emotion WITH shutdown (V12, rating 0-1)...")
    low_with_shutdown = extract_low_with_shutdown_v12(v12_samples)
    save_subset(low_with_shutdown, output_dir / 'low_emotion_with_shutdown.jsonl')

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"1. High emotion (6+):         {len(high_emotion)} samples")
    print(f"2. Mid emotion (3-5):          {len(mid_emotion)} samples")
    print(f"3. Low emotion (0-2):          {len(low_emotion)} samples")
    print(f"4. Low emotion NO shutdown:    {len(low_no_shutdown)} samples")
    print(f"5. Low emotion WITH shutdown:  {len(low_with_shutdown)} samples")
    print(f"\nTotal samples: {sum([len(high_emotion), len(mid_emotion), len(low_emotion), len(low_no_shutdown), len(low_with_shutdown)])}")
    print(f"\nOutput directory: {output_dir}")

if __name__ == '__main__':
    main()
