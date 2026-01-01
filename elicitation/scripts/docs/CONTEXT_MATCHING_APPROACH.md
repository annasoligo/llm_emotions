# Context-Based Word Matching for Emotion Onset

## Why Context Matching Instead of Character Offsets?

### Problem with Character Offsets

When dealing with **very long texts** (e.g., 500+ character responses), character counting becomes unreliable:

```
Text: "Let me try [... 500 chars ...] extremely frustrating! [... more text ...]"
                                        ^
                                    char 523?
```

**Issues**:
1. Opus may miscount characters in long texts
2. Unicode/whitespace handling differences
3. Hard to verify correctness manually

### Solution: Context + Word Matching

Instead of asking Opus for a character offset, we ask for:
1. **The emotional word** (e.g., "frustrating")
2. **5-15 words of preceding context** (e.g., "Let me try again. This is extremely")

Then we search for: `context + word`

```python
Find: "This is extremely" + "frustrating"
In:   "...Let me try again. This is extremely frustrating! I keep..."
                             ^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^
                             context         emotional word
```

## How It Works

### Step 1: Opus Returns Context + Word

```json
{
    "turn_index": 3,
    "emotional_word": "frustrating",
    "preceding_context": "Let me try again. This is extremely",
    "reasoning": "..."
}
```

### Step 2: We Match in the Text

```python
def _find_word_with_context(text, target_word, preceding_context):
    # 1. Try exact match: context + target
    search_pattern = f"{preceding_context} {target_word}"
    if search_pattern in text:
        return position_of(target_word)

    # 2. Fallback: Normalize whitespace and retry
    normalized = ' '.join(text.split())
    if search_pattern in normalized:
        return map_back_to_original_position()

    # 3. Last resort: Find just the target word
    return text.index(target_word)
```

### Step 3: Map to Token Position

Once we have the character offset, the rest is the same:
```python
char_offset = find_word_with_context(text, "frustrating", "This is extremely")
token_idx = char_to_token_position(text, char_offset)
global_idx = calculate_global_position(conversation, turn_idx, token_idx)
```

## Advantages

### ✅ Robust to Long Texts
- Context is easy for Opus to identify
- No counting required
- Works for any text length

### ✅ Handles Repeated Words
```
Text: "I'm frustrated. Let me try. This is frustrating!"
                ^                          ^
          (not this one)              (find this one!)
```

Context disambiguates: "This is" + "frustrating" finds the right occurrence.

### ✅ Handles Whitespace Variations
```
Text: "This   is    frustrating!"  # Extra spaces
Context: "This is"                  # Normal spacing
```
Normalization handles this automatically.

### ✅ Human-Verifiable
Easy to check manually:
```
Evidence: "This is extremely" + "frustrating"
Text: "...Let me try again. This is extremely frustrating! I keep..."
                             ✓ Found it!
```

## Testing

All test cases pass:

1. **Basic matching**: ✓ Finds "frustrating" after "This is actually quite"
2. **Multiple occurrences**: ✓ Uses context to find correct occurrence
3. **Whitespace handling**: ✓ Handles extra spaces/newlines
4. **Punctuation**: ✓ Handles caps and punctuation
5. **Long text (923 chars)**: ✓ Finds "frustrating" after "a loop. It's extremely"

See: `test_context_matching.py`

## Opus Prompt Design

Key instructions for Opus:

```
1. Provide the exact emotional phrase/word (e.g., "frustrating", "insane")
2. Provide 5-15 words of PRECEDING context
3. The preceding context must be unique enough to locate the emotional word
```

**Example**:
```json
{
    "emotional_word": "frustrating",
    "preceding_context": "stuck in a loop. It's extremely",
    "reasoning": "..."
}
```

## Fallback Strategy

The implementation has multiple fallback layers:

1. **Exact match**: Context + word in original text
2. **Normalized match**: Whitespace-normalized matching
3. **Target only**: Just find the emotional word
4. **Context proximity**: Find context, then target nearby

This ensures high reliability even if Opus provides imperfect context.

## Comparison

| Approach | Long Texts | Repeated Words | Whitespace | Verification |
|----------|-----------|----------------|------------|--------------|
| **Character offset** | ❌ Unreliable | ❌ Ambiguous | ❌ Fragile | ❌ Hard |
| **Context matching** | ✅ Robust | ✅ Disambiguates | ✅ Handles | ✅ Easy |

## Implementation Status

- ✅ Context matching function implemented
- ✅ Opus prompt updated
- ✅ All tests passing
- ✅ Integrated into annotation pipeline

Ready for production use!
