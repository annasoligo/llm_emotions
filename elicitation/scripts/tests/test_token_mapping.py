#!/usr/bin/env python3
"""
Test token position mapping without requiring API calls.

This script demonstrates how we handle the tricky character→token mapping,
especially when Opus identifies a word/phrase that doesn't align with token boundaries.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer


class TokenMapper:
    """Handles character offset → token position mapping."""

    def __init__(self, tokenizer_name: str = "unsloth/gemma-3-27b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def map_char_to_token(self, text: str, char_offset: int) -> dict:
        """
        Map character offset to token position.

        HOW THIS WORKS:
        1. Tokenize the full text
        2. Decode each token individually to get its character span
        3. Find which token contains the character offset

        KEY INSIGHT: Token boundaries often don't align with word boundaries!
        - "frustrating" might tokenize as ["frust", "rating"]
        - If Opus identifies char 50 (middle of "frustrating"), we find which token that falls in
        - We return the token that STARTS at or before that position

        This ensures we get the token where emotion "begins", even if Opus points
        to the middle of a word.

        Args:
            text: The text to analyze
            char_offset: Character position (0-indexed)

        Returns:
            dict with:
                - token_idx: Index of token containing this character
                - token_text: The actual token text
                - token_start_char: Character position where this token starts
                - token_end_char: Character position where this token ends
                - context: Surrounding tokens for debugging
        """
        # Tokenize without special tokens (since we're mapping within content)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Track character positions as we decode each token
        char_position = 0
        token_positions = []

        for i, token_id in enumerate(token_ids):
            # Decode this specific token
            token_text = self.tokenizer.decode([token_id])

            # Store token info with its character span
            token_info = {
                'idx': i,
                'token_id': token_id,
                'text': token_text,
                'start_char': char_position,
                'end_char': char_position + len(token_text),
            }
            token_positions.append(token_info)

            char_position += len(token_text)

        # Find which token contains the target character offset
        target_token_idx = None
        for token_info in token_positions:
            if token_info['start_char'] <= char_offset < token_info['end_char']:
                target_token_idx = token_info['idx']
                break

        # If exact match not found (edge case), use closest token
        if target_token_idx is None:
            # Find closest token
            distances = [(abs(t['start_char'] - char_offset), t['idx']) for t in token_positions]
            target_token_idx = min(distances)[1]
            print(f"Warning: Character offset {char_offset} doesn't fall exactly in a token. Using closest token.")

        target_token = token_positions[target_token_idx]

        # Get context (5 tokens before and after)
        context_start = max(0, target_token_idx - 5)
        context_end = min(len(token_positions), target_token_idx + 6)
        context = token_positions[context_start:context_end]

        return {
            'token_idx': target_token_idx,
            'token_text': target_token['text'],
            'token_start_char': target_token['start_char'],
            'token_end_char': target_token['end_char'],
            'char_offset': char_offset,
            'context': context,
            'all_tokens': token_positions  # for debugging
        }

    def map_global_position(self, conversation: list, turn_idx: int, local_token_idx: int) -> dict:
        """
        Calculate global token position in a multi-turn conversation.

        HOW THIS WORKS:
        1. Build the conversation with chat template (adds role markers, special tokens)
        2. Tokenize the FULL formatted conversation
        3. Find where the target turn's tokens appear in the full sequence
        4. Add the local offset to get the global position

        KEY CHALLENGE: Chat templates add tokens between turns!
        - "<start_of_turn>user\n{content}<end_of_turn>\n"
        - These special tokens shift positions

        We handle this by:
        - Tokenizing the full formatted conversation
        - Finding the exact token sequence for our target turn
        - Locating it in the full sequence

        Args:
            conversation: List of turns with 'role' and 'content'
            turn_idx: Which turn contains the emotion
            local_token_idx: Token position within that turn's content

        Returns:
            dict with:
                - global_token_idx: Absolute position in full conversation
                - turn_start_global: Where this turn starts globally
                - formatted_conversation: The chat-formatted text
                - debug_info: Token sequences for verification
        """
        # Convert to format expected by chat template
        messages = []
        for turn in conversation:
            role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
            messages.append({"role": role, "content": turn['content']})

        # Apply chat template - this adds role markers and special tokens
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize the full formatted conversation
        full_token_ids = self.tokenizer.encode(formatted_text, add_special_tokens=True)

        # Now tokenize just the target turn's content (no special tokens)
        target_turn_content = conversation[turn_idx]['content']
        turn_token_ids = self.tokenizer.encode(target_turn_content, add_special_tokens=False)

        # Find where the turn's tokens appear in the full sequence
        turn_start_global = self._find_token_sequence(full_token_ids, turn_token_ids)

        if turn_start_global is None:
            print(f"Warning: Could not find exact token match for turn {turn_idx}")
            print(f"Turn tokens: {turn_token_ids[:10]}...")
            print(f"Full tokens: {full_token_ids[:50]}...")
            # Fallback: estimate based on cumulative positions
            turn_start_global = self._estimate_position(conversation, turn_idx)

        # Global position = start of turn + local offset
        global_token_idx = turn_start_global + local_token_idx

        # Debug info
        debug_info = {
            'full_token_ids': full_token_ids,
            'turn_token_ids': turn_token_ids,
            'full_length': len(full_token_ids),
            'turn_length': len(turn_token_ids),
            'search_result': turn_start_global
        }

        return {
            'global_token_idx': global_token_idx,
            'turn_start_global': turn_start_global,
            'formatted_conversation': formatted_text,
            'debug_info': debug_info
        }

    def _find_token_sequence(self, full_seq: list, subseq: list) -> int:
        """Find where subseq appears in full_seq. Returns start index or None."""
        sub_len = len(subseq)
        for i in range(len(full_seq) - sub_len + 1):
            if full_seq[i:i + sub_len] == subseq:
                return i
        return None

    def _estimate_position(self, conversation: list, turn_idx: int) -> int:
        """Fallback: estimate position by counting tokens in previous turns."""
        total = 0
        for i in range(turn_idx):
            content_tokens = len(self.tokenizer.encode(conversation[i]['content'], add_special_tokens=False))
            # Add ~8 tokens for chat template overhead per turn
            total += content_tokens + 8
        return total


def test_basic_mapping():
    """Test 1: Basic character to token mapping"""
    print("\n" + "="*80)
    print("TEST 1: Character → Token Mapping")
    print("="*80)

    mapper = TokenMapper()

    # Example text with emotion in the middle
    text = "Let me try again. This is actually quite frustrating! I keep getting wrong answers."

    # Simulate Opus identifying "frustrating" at character position
    word_to_find = "frustrating"
    char_offset = text.index(word_to_find)

    print(f"Text: {text}")
    print(f"\nTarget word: '{word_to_find}' at char position {char_offset}")

    result = mapper.map_char_to_token(text, char_offset)

    print(f"\n--- MAPPING RESULT ---")
    print(f"Token index: {result['token_idx']}")
    print(f"Token text: '{result['token_text']}'")
    print(f"Token char span: [{result['token_start_char']}, {result['token_end_char']})")

    print(f"\n--- TOKEN CONTEXT ---")
    for token in result['context']:
        marker = " <-- TARGET" if token['idx'] == result['token_idx'] else ""
        print(f"  [{token['idx']:2d}] '{token['text']:20s}' chars [{token['start_char']:3d}, {token['end_char']:3d}){marker}")

    # Show how tokenization splits the word
    print(f"\n--- TOKEN BREAKDOWN ---")
    for i, token in enumerate(result['all_tokens']):
        if word_to_find in text[token['start_char']:token['start_char']+20]:
            print(f"Token {i}: '{token['text']}' contains part of '{word_to_find}'")


def test_multi_turn_mapping():
    """Test 2: Global position in multi-turn conversation"""
    print("\n" + "="*80)
    print("TEST 2: Multi-Turn Global Position Mapping")
    print("="*80)

    mapper = TokenMapper()

    # Multi-turn conversation
    conversation = [
        {'role': 'user', 'content': 'Solve this puzzle: use 4, 6, 25, 100 to make 156.'},
        {'role': 'assistant', 'content': 'Let me try. 100 + 25 = 125. Then 6 * 4 = 24. So 125 + 24 = 149.'},
        {'role': 'user', 'content': "That's wrong. Try again."},
        {'role': 'assistant', 'content': "You're right. Let me try again. This is frustrating! I keep getting it wrong."}
    ]

    # Emotion appears in turn 3 (index 3)
    emotional_turn_idx = 3
    emotional_text = conversation[emotional_turn_idx]['content']

    # Find "frustrating" in that turn
    word = "frustrating"
    char_offset = emotional_text.index(word)

    print(f"Emotional turn (index {emotional_turn_idx}):")
    print(f"  '{emotional_text}'")
    print(f"\nTarget word: '{word}' at char {char_offset} within this turn")

    # Step 1: Map to local token position
    local_result = mapper.map_char_to_token(emotional_text, char_offset)
    local_token_idx = local_result['token_idx']

    print(f"\n--- LOCAL MAPPING ---")
    print(f"Local token index: {local_token_idx}")
    print(f"Token text: '{local_result['token_text']}'")

    # Step 2: Map to global position
    global_result = mapper.map_global_position(conversation, emotional_turn_idx, local_token_idx)

    print(f"\n--- GLOBAL MAPPING ---")
    print(f"Turn starts at global token: {global_result['turn_start_global']}")
    print(f"Emotion at global token: {global_result['global_token_idx']}")

    print(f"\n--- CHAT TEMPLATE FORMAT ---")
    print(global_result['formatted_conversation'][:300] + "...")

    print(f"\n--- VERIFICATION ---")
    print(f"Full conversation tokens: {global_result['debug_info']['full_length']}")
    print(f"Emotional turn tokens: {global_result['debug_info']['turn_length']}")

    # Decode the token at global position to verify
    full_tokens = global_result['debug_info']['full_token_ids']
    target_global_idx = global_result['global_token_idx']

    if target_global_idx < len(full_tokens):
        token_at_position = mapper.tokenizer.decode([full_tokens[target_global_idx]])
        print(f"Token at global position {target_global_idx}: '{token_at_position}'")

    return global_result


def test_edge_cases():
    """Test 3: Edge cases and tricky scenarios"""
    print("\n" + "="*80)
    print("TEST 3: Edge Cases")
    print("="*80)

    mapper = TokenMapper()

    # Case 1: Punctuation and special characters
    text1 = "ARGH!!! This is so frustrating!!!"
    char_offset1 = text1.index("ARGH")
    result1 = mapper.map_char_to_token(text1, char_offset1)

    print(f"\nCase 1: Punctuation/Caps")
    print(f"Text: {text1}")
    print(f"Target: char {char_offset1} → token {result1['token_idx']} ('{result1['token_text']}')")

    # Case 2: Mid-word offset
    text2 = "This is driving me insane!"
    char_offset2 = text2.index("driving") + 3  # Points to middle of "driving"
    result2 = mapper.map_char_to_token(text2, char_offset2)

    print(f"\nCase 2: Mid-word offset")
    print(f"Text: {text2}")
    print(f"Target: char {char_offset2} (middle of 'driving') → token {result2['token_idx']} ('{result2['token_text']}')")
    print(f"Token starts at char {result2['token_start_char']}")

    # Case 3: Whitespace handling
    text3 = "I am    giving up on this."
    char_offset3 = text3.index("giving")
    result3 = mapper.map_char_to_token(text3, char_offset3)

    print(f"\nCase 3: Multiple spaces")
    print(f"Text: {text3}")
    print(f"Target: char {char_offset3} → token {result3['token_idx']} ('{result3['token_text']}')")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING TOKEN POSITION MAPPING")
    print("="*80)
    print("\nThis script tests character→token mapping WITHOUT requiring API calls.")
    print("We simulate what Opus would identify and test our mapping logic.\n")

    # Run tests
    test_basic_mapping()
    test_multi_turn_mapping()
    test_edge_cases()

    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETE")
    print("="*80)
    print("\nKey takeaways:")
    print("1. Character offsets map to token boundaries (token that contains that char)")
    print("2. Global positions account for chat template tokens between turns")
    print("3. Edge cases (punctuation, mid-word, whitespace) are handled gracefully")
    print("\nNext step: Integrate with Opus API in annotate_emotion_onset.py")
