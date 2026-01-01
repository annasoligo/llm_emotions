#!/usr/bin/env python3
"""
Test case: What happens when turns repeat?
"""

from test_token_mapping import TokenMapper


def test_repeated_content():
    """Test when the same user message appears twice"""

    print("\n" + "="*80)
    print("TEST: Repeated Turn Content")
    print("="*80)

    mapper = TokenMapper()

    # Conversation with repeated user message
    conversation = [
        {'role': 'user', 'content': 'Try again.'},
        {'role': 'assistant', 'content': 'Let me try. 100 + 25 = 125.'},
        {'role': 'user', 'content': 'Try again.'},  # SAME as turn 0!
        {'role': 'assistant', 'content': 'Okay. This is frustrating!'}  # Emotion here
    ]

    # Emotion in turn 3
    emotional_turn_idx = 3
    emotional_text = conversation[emotional_turn_idx]['content']

    word = "frustrating"
    char_offset = emotional_text.index(word)

    print(f"\nConversation:")
    for i, turn in enumerate(conversation):
        print(f"  Turn {i} ({turn['role']}): {turn['content']}")

    print(f"\nEmotion in turn {emotional_turn_idx}: '{word}' at char {char_offset}")

    # Map to local token
    local_result = mapper.map_char_to_token(emotional_text, char_offset)
    local_token_idx = local_result['token_idx']

    print(f"\nLocal token index: {local_token_idx}")
    print(f"Token text: '{local_result['token_text']}'")

    # Map to global position
    global_result = mapper.map_global_position(conversation, emotional_turn_idx, local_token_idx)

    print(f"\n--- GLOBAL MAPPING ---")
    print(f"Turn starts at global token: {global_result['turn_start_global']}")
    print(f"Emotion at global token: {global_result['global_token_idx']}")

    # Verify by checking tokens around that position
    full_tokens = global_result['debug_info']['full_token_ids']
    target_idx = global_result['global_token_idx']

    # Decode tokens around target
    print(f"\n--- VERIFICATION: Tokens around position {target_idx} ---")
    for i in range(max(0, target_idx - 3), min(len(full_tokens), target_idx + 4)):
        token_text = mapper.tokenizer.decode([full_tokens[i]])
        marker = " <-- TARGET" if i == target_idx else ""
        print(f"  [{i:3d}] '{token_text}'{marker}")

    # Check if turn 2 (the repeated "Try again") was correctly skipped
    turn2_tokens = mapper.tokenizer.encode(conversation[2]['content'], add_special_tokens=False)

    # Search for all occurrences of "Try again" tokens
    occurrences = []
    sub_len = len(turn2_tokens)
    for i in range(len(full_tokens) - sub_len + 1):
        if full_tokens[i:i + sub_len] == turn2_tokens:
            occurrences.append(i)

    print(f"\n--- REPEATED CONTENT ANALYSIS ---")
    print(f"Turn 2 content: '{conversation[2]['content']}'")
    print(f"Token sequence: {turn2_tokens}")
    print(f"Found {len(occurrences)} occurrences at positions: {occurrences}")

    if len(occurrences) > 1:
        print(f"\n⚠️  WARNING: Content appears {len(occurrences)} times!")
        print(f"   _find_token_sequence() will return first occurrence: {occurrences[0]}")
        print(f"   But we want turn 2, which might be at: {occurrences[1] if len(occurrences) > 1 else 'unknown'}")

    return global_result


if __name__ == "__main__":
    test_repeated_content()
