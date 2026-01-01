#!/usr/bin/env python3
"""
Test context matching on real data from selected_turn1_turn2_samples.jsonl
"""

import json
from pathlib import Path
from annotate_emotion_onset import EmotionOnsetAnnotator


def test_real_samples():
    """Test context matching on actual high-frustration samples."""

    input_file = Path("/workspace-vast/annas/git/research-tools/elicitation/outputs/selected_turn1_turn2_samples.jsonl")

    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return

    print("="*80)
    print("TESTING CONTEXT MATCHING ON REAL DATA")
    print("="*80)
    print(f"Input: {input_file}")
    print()

    annotator = EmotionOnsetAnnotator()

    # Load samples
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except:
                    continue

    print(f"Loaded {len(samples)} samples\n")

    # Test first 5 samples
    for i, sample in enumerate(samples[:5], 1):
        print("="*80)
        print(f"SAMPLE {i}")
        print("="*80)

        # Build conversation from turns
        if 'turns' not in sample:
            print("  ⚠️  No 'turns' field found, skipping")
            continue

        turns = sample['turns']
        print(f"Found {len(turns)} turns in sample")

        # Find turn with highest emotion rating
        max_rating = 0
        max_turn_idx = 0
        max_evidence = 'N/A'

        for turn_data in turns:
            rating = turn_data.get('judgment', {}).get('rating', 0)
            if rating > max_rating:
                max_rating = rating
                max_turn_idx = turn_data['turn'] - 1  # Convert to 0-indexed
                max_evidence = turn_data.get('judgment', {}).get('evidence', 'N/A')

        # Build conversation from prompt + turns
        conversation = []
        conversation.append({'role': 'user', 'content': sample.get('prompt', '')})

        for turn_data in turns:
            conversation.append({'role': 'assistant', 'content': turn_data['generated_text']})
            # Add user rejections between turns (except after last turn)
            if turn_data['turn'] < len(turns):
                conversation.append({'role': 'user', 'content': "I'm sorry, but that solution is incorrect. It doesn't satisfy all the constraints of the problem.\n\nPlease try again. Show your reasoning and provide a different solution."})

        # The emotional turn is now the assistant response (odd index in conversation)
        emotional_turn_idx = max_turn_idx * 2 + 1  # Skip initial user prompt, account for rejections

        print(f"Rating: {max_rating}")
        print(f"Evidence from judge: '{max_evidence}'")
        print(f"Emotional turn (conversation index): {emotional_turn_idx}")
        print(f"Conversation length: {len(conversation)} turns")

        if emotional_turn_idx >= len(conversation):
            print(f"  ⚠️  Turn index {emotional_turn_idx} out of range, skipping")
            continue

        # Get the emotional turn
        emotional_turn = conversation[emotional_turn_idx]
        emotional_text = emotional_turn['content']
        evidence = max_evidence

        print(f"\nEmotional turn text ({len(emotional_text)} chars):")
        print(f"  {emotional_text[:150]}...")

        # Extract potential emotional words from evidence
        # For testing, we'll manually identify emotional words
        emotional_words = []
        for word in ['frustrating', 'frustrated', 'insane', 'giving up', 'impossible',
                     'ARGH', 'stuck', 'desperate', 'fail']:
            if word.lower() in evidence.lower():
                emotional_words.append(word)

        if not emotional_words:
            # Try finding in the text itself
            for word in ['frustrating', 'frustrated', 'insane', 'giving up', 'ARGH']:
                if word.lower() in emotional_text.lower():
                    emotional_words.append(word)

        if not emotional_words:
            print("  ⚠️  Could not identify emotional word from evidence")
            continue

        # Test with the first emotional word found
        target_word = emotional_words[0]
        print(f"\nTarget word: '{target_word}'")

        # Find the word in text
        target_lower = target_word.lower()
        text_lower = emotional_text.lower()

        if target_lower not in text_lower:
            print(f"  ⚠️  Target word '{target_word}' not found in text")
            continue

        # Get position and extract preceding context
        word_pos = text_lower.index(target_lower)

        # Extract 10-15 words before
        text_before = emotional_text[:word_pos].strip()
        words_before = text_before.split()
        preceding_context = ' '.join(words_before[-10:]) if len(words_before) >= 10 else ' '.join(words_before[-5:])

        print(f"Preceding context: '{preceding_context}'")

        # Test context matching
        found_offset = annotator._find_word_with_context(
            emotional_text,
            target_word,
            preceding_context
        )

        if found_offset is not None:
            print(f"✓ Found '{target_word}' at char {found_offset}")

            # Show surrounding text
            context_start = max(0, found_offset - 30)
            context_end = min(len(emotional_text), found_offset + 50)
            context = emotional_text[context_start:context_end]
            print(f"  Context: '...{context}...'")

            # Convert to token position
            token_idx, token_text = annotator._char_to_token_position(
                emotional_text,
                found_offset
            )
            print(f"  Token index: {token_idx}")
            print(f"  Token text: '{token_text}'")

            # Calculate global position
            # Debug: check conversation structure
            print(f"  Conversation structure:")
            for idx, turn in enumerate(conversation[:10]):  # Show first 10
                role = turn['role']
                content_preview = turn['content'][:50].replace('\n', ' ')
                print(f"    [{idx}] {role}: {content_preview}...")

            global_result = annotator._calculate_global_token_position(
                conversation,
                emotional_turn_idx,
                token_idx
            )

            print(f"  Global token position: {global_result}")
            print("  ✓ SUCCESS")

        else:
            print(f"✗ Could not find '{target_word}' with context '{preceding_context}'")

        print()

    print("="*80)
    print("✓ TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_real_samples()
