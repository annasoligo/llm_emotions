# Token Position Mapping: How It Works

## The Challenge

When Opus identifies where emotion first appears, it provides:
- A text quote (e.g., "This is frustrating!")
- A character offset (e.g., position 342 in the text)

But we need:
- A **token index** for extracting activations from the model

**Why is this hard?**
1. **Tokenization doesn't align with words**: "frustrating" might tokenize as ["frust", "rating"]
2. **Chat templates add tokens**: Special tokens between turns shift positions
3. **Mid-word offsets**: Opus might point to the middle of a word

## Our Solution

### Step 1: Character → Token Mapping (Local Position)

```python
def map_char_to_token(text, char_offset):
    # Tokenize the text
    tokens = tokenizer.encode(text)

    # Decode each token to find its character span
    char_pos = 0
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        token_start = char_pos
        token_end = char_pos + len(token_text)

        # Check if this token contains the character offset
        if token_start <= char_offset < token_end:
            return i  # This is our token!

        char_pos = token_end
```

**Key insight**: We find the token whose character span **contains** the target offset.

### Example 1: Basic Case

```
Text: "Let me try again. This is frustrating!"
Opus identifies: "frustrating" at char 27

Tokenization:
[0] "Let"        chars [0, 3)
[1] " me"       chars [3, 6)
[2] " try"      chars [6, 10)
[3] " again"    chars [10, 16)
[4] "."         chars [16, 17)
[5] " This"     chars [17, 22)
[6] " is"       chars [22, 25)
[7] " frust"    chars [25, 31)  ← CONTAINS char 27!
[8] "rating"    chars [31, 37)
[9] "!"         chars [37, 38)

Result: Token index 7 ("frust")
```

**Note**: Even though Opus identified "frustrating", we return token 7 ("frust") because that's where the emotional content **starts**.

### Example 2: Mid-Word Offset

```
Text: "This is driving me insane!"
Opus identifies: char 15 (middle of "driving")

Tokenization:
[0] "This"      chars [0, 4)
[1] " is"       chars [4, 7)
[2] " driv"     chars [7, 12)   ← Contains char 15? No
[3] "ing"       chars [12, 15)  ← Contains char 15? No (15 is right at boundary)
[4] " me"       chars [15, 18)  ← Contains char 15? Yes!

Result: Token index 4 (" me")
```

Wait, that's wrong! The issue is char 15 falls at a token boundary.

**Refined logic**: If char_offset is exactly at a boundary, we take the token that **starts** there (not the previous one).

```python
if token_start <= char_offset < token_end:
    return i
```

With this logic:
- char 15 falls in token 4 (" me") since 15 >= 15 and 15 < 18

But we actually want token 3 ("ing") since that's part of "driving"!

**Better approach**: When char_offset points to "driving", we should look at which token contains the **start** of that word, not a random char in the middle.

**Solution**: Opus should provide the character offset of the **start of the emotional word/phrase**, not a random position in the middle.

### Step 2: Local → Global Token Position

Once we have the local token index within a turn, we need to find its position in the full conversation.

**Challenge**: Chat templates add tokens between turns!

```
User turn:    "Solve this puzzle"
             ↓ Chat template adds:
             <start_of_turn>user\n{content}<end_of_turn>\n