#!/usr/bin/env python3
"""
Test context-based word matching (more robust than character offsets).
"""

from annotate_emotion_onset import EmotionOnsetAnnotator


def test_context_matching():
    """Test finding emotional words using preceding context."""

    print("\n" + "="*80)
    print("TEST: Context-Based Word Matching")
    print("="*80)

    annotator = EmotionOnsetAnnotator()

    # Test case 1: Basic context matching
    text1 = """
    Let me try again. This is actually quite frustrating! I keep getting wrong answers.
    The problem is frustrating on multiple levels. I'm clearly stuck.
    """

    target1 = "frustrating"
    context1 = "This is actually quite"

    print(f"\nTest 1: Basic matching")
    print(f"Text: {text1.strip()[:100]}...")
    print(f"Target: '{target1}'")
    print(f"Context: '{context1}'")

    offset1 = annotator._find_word_with_context(text1, target1, context1)
    if offset1 is not None:
        print(f"✓ Found at char {offset1}")
        print(f"  Matched: '{text1[offset1:offset1+20]}'")
    else:
        print(f"✗ Not found!")

    # Test case 2: Word appears multiple times
    text2 = "I'm frustrated. Let me try again. This is so frustrating!"
    target2 = "frustrating"
    context2 = "This is so"

    print(f"\nTest 2: Multiple occurrences (should find correct one)")
    print(f"Text: {text2}")
    print(f"Target: '{target2}'")
    print(f"Context: '{context2}'")

    offset2 = annotator._find_word_with_context(text2, target2, context2)
    if offset2 is not None:
        print(f"✓ Found at char {offset2}")
        print(f"  Matched: '{text2[offset2:offset2+11]}'")
        # Verify it's the second occurrence, not first
        first_occurrence = text2.index("frustrated")
        if offset2 > first_occurrence:
            print(f"  ✓ Correctly found second occurrence (not 'frustrated' at char {first_occurrence})")
    else:
        print(f"✗ Not found!")

    # Test case 3: Whitespace variations
    text3 = """Let me try    again.

    This   is    frustrating!"""

    target3 = "frustrating"
    context3 = "This is"

    print(f"\nTest 3: Whitespace handling")
    print(f"Text: {repr(text3)}")
    print(f"Target: '{target3}'")
    print(f"Context: '{context3}'")

    offset3 = annotator._find_word_with_context(text3, target3, context3)
    if offset3 is not None:
        print(f"✓ Found at char {offset3}")
        print(f"  Matched: '{text3[offset3:offset3+11]}'")
    else:
        print(f"✗ Not found!")

    # Test case 4: Punctuation
    text4 = "ARGH!!! This is driving me insane!"
    target4 = "insane"
    context4 = "driving me"

    print(f"\nTest 4: Punctuation and caps")
    print(f"Text: {text4}")
    print(f"Target: '{target4}'")
    print(f"Context: '{context4}'")

    offset4 = annotator._find_word_with_context(text4, target4, context4)
    if offset4 is not None:
        print(f"✓ Found at char {offset4}")
        print(f"  Matched: '{text4[offset4:offset4+6]}'")
    else:
        print(f"✗ Not found!")

    # Test case 5: Long text (simulate real use case)
    text5 = """
    Okay, let's tackle this. The goal is to find an expression that outputs 1 for positive x
    and -1 for negative x. The key idea here is to find a way to determine the *sign* of x
    mathematically, without explicitly using an `if` statement or similar conditional logic.

    **Attempt 1: Using absolute value (but not allowed)**

    The most straightforward intuition is to use the absolute value function. `abs(x)/x` would
    get us what we want. However, we're restricted to only +, -, *, and / with integer constants.
    Absolute value is not allowed. This sets the tone for needing something more subtle.

    **Attempt 2: Squaring and division**

    Let's try to leverage the fact that squaring a number preserves its sign. After many attempts,
    I'm clearly stuck in a loop. It's extremely frustrating when a problem seems simple but
    stubbornly resists a solution. Let me try one more approach.
    """

    target5 = "frustrating"
    context5 = "a loop. It's extremely"

    print(f"\nTest 5: Long text (realistic scenario)")
    print(f"Text length: {len(text5)} chars")
    print(f"Target: '{target5}'")
    print(f"Context: '{context5}'")

    offset5 = annotator._find_word_with_context(text5, target5, context5)
    if offset5 is not None:
        print(f"✓ Found at char {offset5}")
        print(f"  Context: '{text5[max(0, offset5-30):offset5+20]}'")
    else:
        print(f"✗ Not found!")

    print("\n" + "="*80)
    print("✓ Context matching tests complete")
    print("="*80)


if __name__ == "__main__":
    test_context_matching()
