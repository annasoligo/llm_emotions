# Known Limitations and Edge Cases

## Token Position Mapping

### ⚠️ Limitation: Repeated Turn Content

**Issue**: If multiple turns have identical content, `_find_token_sequence()` returns the first occurrence.

**Example**:
```python
Turn 0: "Try again."
Turn 1: "Let me try..."
Turn 2: "Try again."  # Same as turn 0!
Turn 3: "Frustrated!"
```

If we search for turn 2's token sequence, we'll find turn 0 instead.

**Impact on Our Use Case**:
- **LOW RISK** for emotional turns
  - Emotional responses are unlikely to be exact duplicates
  - Each emotional expression tends to be unique

- **MEDIUM RISK** for baseline extraction
  - First user turn (baseline window) might repeat
  - But we explicitly target turn 0, so no ambiguity

**Workaround** (if needed):
Search from an estimated starting position based on turns already processed:

```python
def _find_token_sequence_from_position(self, full_seq, subseq, start_from=0):
    """Find token sequence starting search from a specific position."""
    sub_len = len(subseq)
    for i in range(start_from, len(full_seq) - sub_len + 1):
        if full_seq[i:i + sub_len] == subseq:
            return i
    return None
```

Then estimate starting position:
```python
# Estimate where this turn should start (after previous turns)
estimated_start = sum(len(tokenizer.encode(conv[i]['content'])) + 8
                     for i in range(turn_idx))

# Search from there
turn_start = _find_token_sequence_from_position(
    full_tokens, turn_tokens, start_from=estimated_start - 20  # some buffer
)
```

**Recommendation**:
- Current implementation is **sufficient** for this experiment
- If we encounter issues, add the position-based search

### Character-to-Token Mapping Edge Cases

**Handled correctly**:
- ✅ Mid-word character offsets
- ✅ Punctuation and special characters
- ✅ Multiple spaces
- ✅ Words split across tokens

**Assumption**:
- Opus provides character offset at the **start** of emotional content
- If Opus points mid-word, we return the token containing that position

## Chat Template Variations

**Issue**: Different models use different chat templates.

**Current implementation**:
- Hardcoded for Gemma-3-27b format:
  ```
  <bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n...
  ```

**Impact**:
- Works correctly for Gemma
- Would need adjustment for other models (GPT, Claude, Llama, etc.)

**Solution if needed**:
- Pass tokenizer to all functions
- Use `tokenizer.apply_chat_template()` consistently
- No hardcoded format assumptions

## Recommendations for Production Use

If scaling this to many samples:

1. **Add validation**:
   - Check that decoded token at global position matches expected text
   - Alert if mismatch detected

2. **Add robustness**:
   - Use position-based search for repeated content
   - Add fuzzy matching if exact sequence not found

3. **Add logging**:
   - Log all ambiguous cases
   - Save token maps for manual review

4. **Test edge cases**:
   - Very long turns (>512 tokens)
   - Special characters / emojis
   - Code blocks in responses
