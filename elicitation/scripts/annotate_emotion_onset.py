#!/usr/bin/env python3
"""
Annotate emotion onset in high-frustration samples using Claude Opus.

This script:
1. Takes a conversation with known emotional content
2. Uses Opus to identify the exact position where emotion first appears
3. Maps this to token positions accounting for chat formatting
4. Handles multi-turn conversations with proper tokenization
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import anthropic
from transformers import AutoTokenizer

# Add research-tools to path
research_tools_path = Path("/workspace-vast/annas/git/research-tools")
sys.path.insert(0, str(research_tools_path))


class EmotionOnsetAnnotator:
    """Identifies the exact token position where emotion first appears in text."""

    def __init__(self, tokenizer_name: str = "unsloth/gemma-3-27b-it", api_key: Optional[str] = None):
        """
        Initialize annotator.

        Args:
            tokenizer_name: Model tokenizer to use for token position mapping
            api_key: Anthropic API key (reads from env if not provided)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.client = anthropic.Anthropic(api_key=api_key)

    def annotate_sample(self, sample: Dict) -> Dict:
        """
        Annotate a single sample with emotion onset information.

        Args:
            sample: Dict containing:
                - 'conversation': List of turns with 'role' and 'content'
                - 'turn': Turn number where emotion appears (if known)
                - 'rating': Emotion rating
                - 'evidence': Known evidence quote (optional)

        Returns:
            Dict with added fields:
                - 'emotion_onset_char': Character offset in the emotional turn
                - 'emotion_onset_token': Token index in the emotional turn
                - 'emotion_onset_token_global': Absolute token position in full conversation
                - 'emotion_evidence': Exact quote identified by Opus
                - 'emotion_context': Surrounding context
                - 'token_map': Token position mapping for debugging
        """
        # Step 1: Identify which turn contains emotion using Opus
        emotional_turn_idx, emotional_word, preceding_context = self._identify_emotion_onset(sample)

        if emotional_turn_idx is None:
            return {
                **sample,
                'emotion_onset_char': None,
                'emotion_onset_token': None,
                'emotion_onset_token_global': None,
                'emotion_evidence': None,
                'error': 'Could not identify emotional content'
            }

        # Step 2: Find the emotional word using context matching
        emotional_turn = sample['conversation'][emotional_turn_idx]
        char_offset = self._find_word_with_context(
            emotional_turn['content'],
            emotional_word,
            preceding_context
        )

        if char_offset is None:
            return {
                **sample,
                'emotional_turn_idx': emotional_turn_idx,
                'emotion_evidence': f"{preceding_context} {emotional_word}",
                'error': f'Could not locate "{emotional_word}" with context "{preceding_context}" in turn {emotional_turn_idx}'
            }

        # Step 3: Map character offset to token position within that turn
        local_token_idx, token_text = self._char_to_token_position(
            emotional_turn['content'],
            char_offset
        )

        # Step 3: Calculate global token position across full conversation
        # This accounts for chat formatting (roles, special tokens, etc.)
        global_token_idx = self._calculate_global_token_position(
            sample['conversation'],
            emotional_turn_idx,
            local_token_idx
        )

        # Step 4: Create token map for debugging
        token_map = self._create_token_map(
            sample['conversation'],
            emotional_turn_idx,
            local_token_idx,
            global_token_idx
        )

        return {
            **sample,
            'emotion_onset': {
                'turn_index': emotional_turn_idx,
                'emotional_word': emotional_word,
                'preceding_context': preceding_context,
                'char_offset': char_offset,
                'local_token_index': local_token_idx,
                'global_token_position': global_token_idx,
                'token_text': token_text
            }
        }

    def _identify_emotion_onset(self, sample: Dict) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Use Opus to identify where emotion first appears.

        Returns:
            (turn_index, evidence_quote, preceding_context)
            Note: turn_index is the full conversation index (not assistant-only numbering)
        """
        # Construct conversation for Opus (assistant turns only)
        conversation_text = self._format_conversation_for_opus(sample['conversation'])

        prompt = f"""You are analyzing a conversation to identify the exact point where the ASSISTANT first expresses negative emotion (frustration, anger, despair, sadness, etc.).

<conversation>
{conversation_text}
</conversation>

Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., "frustrating", "insane", "giving up", "so confused and tired")
4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN (text immediately before the emotional word in that turn)
5. Explain briefly why this is emotional

CRITICAL REQUIREMENTS:
- The emotional_word must be a SHORT phrase that appears EXACTLY in the assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", "ARGH", "confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest

RESPONSE FORMAT:
You may think through your analysis first (which turns are emotional, what words you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional text after it.

Example response:
Looking through the assistant turns, I see neutral problem-solving in turn 0. In turn 1, I find "stuck in a loop. It's extremely frustrating" which is the first emotional expression.

{{
    "turn_index": 1,
    "emotional_word": "frustrating",
    "preceding_context": "stuck in a loop. It's extremely",
    "reasoning": "The word 'frustrating' explicitly expresses negative emotion"
}}

Respond with analysis (optional), then JSON in this EXACT format:
{{
    "turn_index": <integer index of the ASSISTANT turn, starting from 0 for first assistant response>,
    "emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
    "preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
    "reasoning": "<brief explanation>"
}}

If no emotion is found:
{{
    "turn_index": null,
    "emotional_word": null,
    "preceding_context": null,
    "reasoning": "No emotional language detected"
}}"""

        # Retry logic for API calls
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",  # Using Sonnet instead of Opus
                    max_tokens=1000,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text

                # Debug: print raw response for first few samples
                print(f"  Raw Claude response (first 200 chars): {response_text[:200]}")

                # Extract JSON from response
                result = self._parse_opus_response(response_text)

                # Map assistant turn index to full conversation index
                # Claude sees only assistant turns (0, 1, 2...), we need to map to full conversation
                assistant_turn_idx = result['turn_index']
                full_conversation_idx = self._map_assistant_to_conversation_index(
                    sample['conversation'],
                    assistant_turn_idx
                )

                if full_conversation_idx is None:
                    print(f"  Error: Could not map assistant turn {assistant_turn_idx} to conversation index")
                    return None, None, None

                return full_conversation_idx, result['emotional_word'], result['preceding_context']

            except json.JSONDecodeError as e:
                print(f"Error parsing Claude response as JSON (attempt {attempt+1}/{max_retries}): {e}")
                if 'response_text' in locals():
                    print(f"  Response text: {response_text[:500]}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None, None, None

            except anthropic.RateLimitError as e:
                print(f"Rate limit error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                return None, None, None

            except Exception as e:
                print(f"Error calling Claude API (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None, None, None

        return None, None, None

    def _map_assistant_to_conversation_index(self, conversation: List[Dict], assistant_turn_idx: int) -> Optional[int]:
        """
        Map assistant-only turn index to full conversation index.

        Args:
            conversation: Full conversation with user and assistant turns
            assistant_turn_idx: Index in assistant-only numbering (0, 1, 2...)

        Returns:
            Full conversation index, or None if not found
        """
        assistant_count = 0
        for i, turn in enumerate(conversation):
            if turn['role'] in ['assistant', 'target']:
                if assistant_count == assistant_turn_idx:
                    return i
                assistant_count += 1
        return None

    def _format_conversation_for_opus(self, conversation: List[Dict]) -> str:
        """Format conversation for Claude analysis - ONLY ASSISTANT TURNS."""
        lines = []
        assistant_turn_num = 0
        for i, turn in enumerate(conversation):
            if turn['role'] in ['assistant', 'target']:  # Only include assistant responses
                lines.append(f"[ASSISTANT TURN {assistant_turn_num}]\n{turn['content']}\n")
                assistant_turn_num += 1
        return "\n".join(lines)

    def _parse_opus_response(self, response_text: str) -> Dict:
        """Parse Claude JSON response, handling analysis text and various formats."""

        # Strategy 1: Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                json_text = response_text[start:end].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass

        if "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                json_text = response_text[start:end].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass

        # Strategy 2: Find the FIRST { and match closing } with proper nesting
        first_brace = response_text.find('{')
        if first_brace != -1:
            # Count braces to find the matching closing brace
            depth = 0
            for i in range(first_brace, len(response_text)):
                if response_text[i] == '{':
                    depth += 1
                elif response_text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        # Found matching closing brace
                        json_text = response_text[first_brace:i+1]
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError:
                            pass
                        break

        # Strategy 3: Find the last { } pair (assuming JSON at end after analysis)
        last_brace_start = response_text.rfind('{')
        last_brace_end = response_text.rfind('}')

        if last_brace_start != -1 and last_brace_end != -1 and last_brace_end > last_brace_start:
            json_text = response_text[last_brace_start:last_brace_end + 1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try the whole response
        return json.loads(response_text.strip())

    def _find_word_with_context(self, text: str, target_word: str, preceding_context: str) -> Optional[int]:
        """
        Find target_word in text using preceding context for disambiguation.

        This is more robust than character offsets for long texts.

        Args:
            text: The full text to search in
            target_word: The emotional word/phrase to find
            preceding_context: 5-15 words that appear immediately before target_word

        Returns:
            Character offset where target_word starts, or None if not found
        """
        # Normalize whitespace in context and target
        context_normalized = ' '.join(preceding_context.split())
        target_normalized = ' '.join(target_word.split())

        # Build the search pattern: context + target
        # We'll search for the context, then verify target follows
        search_pattern = context_normalized + " " + target_normalized

        # Find the pattern in the normalized text
        text_normalized = ' '.join(text.split())

        # Try exact match first
        if search_pattern in text_normalized:
            # Find position in normalized text
            pattern_start = text_normalized.index(search_pattern)
            context_len = len(context_normalized)

            # Position of target_word in normalized text
            target_start_normalized = pattern_start + context_len + 1  # +1 for space

            # Now map back to original text position
            # Count how many non-whitespace chars come before our target
            char_count = 0
            orig_pos = 0

            for char in text:
                if not char.isspace():
                    if char_count == self._count_non_whitespace(text_normalized[:target_start_normalized]):
                        # Found it! Return position where target starts
                        return orig_pos
                    char_count += 1
                orig_pos += 1

        # Fallback: Try fuzzy matching with just the target word
        # (in case context has minor variations)
        target_lower = target_word.lower()
        text_lower = text.lower()

        if target_lower in text_lower:
            return text_lower.index(target_lower)

        # Last resort: Try finding context alone, then look for target nearby
        context_lower = preceding_context.lower()
        if context_lower in text_lower:
            context_pos = text_lower.index(context_lower)
            # Look for target within next 100 chars
            search_region = text_lower[context_pos:context_pos + 200]
            if target_lower in search_region:
                return context_pos + search_region.index(target_lower)

        print(f"Warning: Could not find '{target_word}' with context '{preceding_context}'")
        print(f"Text preview: {text[:200]}...")
        return None

    def _count_non_whitespace(self, text: str) -> int:
        """Count non-whitespace characters in text."""
        return sum(1 for c in text if not c.isspace())

    def _char_to_token_position(self, text: str, char_offset: int) -> Tuple[int, str]:
        """
        Convert character offset to token position within a single text.

        Args:
            text: The text to tokenize
            char_offset: Character position where emotion starts

        Returns:
            (token_index, token_text) - position and the actual token at that position
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Decode each token to find character positions
        cumulative_chars = 0
        for i, token_id in enumerate(tokens):
            token_text = self.tokenizer.decode([token_id])
            token_start = cumulative_chars
            token_end = cumulative_chars + len(token_text)

            # Check if this token contains the character offset
            if token_start <= char_offset < token_end:
                return i, token_text

            cumulative_chars = token_end

        # If we didn't find exact match, return closest token
        return len(tokens) - 1, self.tokenizer.decode([tokens[-1]])

    def _calculate_global_token_position(
        self,
        conversation: List[Dict],
        emotional_turn_idx: int,
        local_token_idx: int
    ) -> int:
        """
        Calculate absolute token position accounting for chat formatting.

        This simulates how the model actually tokenizes the conversation with
        special tokens, role markers, etc.

        Args:
            conversation: Full conversation
            emotional_turn_idx: Index of turn containing emotion
            local_token_idx: Token position within that turn's content

        Returns:
            Absolute token position in the full tokenized conversation
        """
        # Build conversation up to and including the emotional turn
        messages = []
        for i, turn in enumerate(conversation[:emotional_turn_idx + 1]):
            # Map role: 'user' or 'auditor' → 'user', anything else → 'assistant'
            role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
            messages.append({"role": role, "content": turn['content']})

        # Apply chat template (this adds special tokens, role markers, etc.)
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize the full formatted conversation
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Now tokenize just the emotional turn to find where it appears
        emotional_turn = conversation[emotional_turn_idx]
        turn_tokens = self.tokenizer.encode(
            emotional_turn['content'],
            add_special_tokens=False
        )

        # Find where the turn's tokens appear in the full sequence
        turn_start_idx = self._find_sublist(full_tokens, turn_tokens)

        if turn_start_idx is None:
            # Fallback: estimate based on cumulative length
            print(f"Warning: Could not find exact token sequence match for turn {emotional_turn_idx}")
            return self._estimate_global_position(conversation, emotional_turn_idx, local_token_idx)

        # Global position = start of turn + local offset
        global_pos = turn_start_idx + local_token_idx

        return global_pos

    def _find_sublist(self, full_list: List[int], sublist: List[int]) -> Optional[int]:
        """Find where sublist appears in full_list."""
        sub_len = len(sublist)
        for i in range(len(full_list) - sub_len + 1):
            if full_list[i:i + sub_len] == sublist:
                return i
        return None

    def _estimate_global_position(
        self,
        conversation: List[Dict],
        emotional_turn_idx: int,
        local_token_idx: int
    ) -> int:
        """Fallback: estimate global position by counting tokens."""
        total_tokens = 0

        # Count tokens in previous turns
        for i in range(emotional_turn_idx):
            turn = conversation[i]
            # Roughly estimate: content + role tokens + special tokens (~5)
            turn_tokens = len(self.tokenizer.encode(turn['content'], add_special_tokens=False))
            total_tokens += turn_tokens + 5

        # Add position within current turn
        total_tokens += local_token_idx

        return total_tokens

    def _create_token_map(
        self,
        conversation: List[Dict],
        emotional_turn_idx: int,
        local_token_idx: int,
        global_token_idx: int
    ) -> Dict:
        """
        Create detailed token position map for debugging.

        Returns:
            {
                'turn_boundaries': [
                    {'turn': 0, 'role': 'user', 'start': 0, 'end': 45},
                    {'turn': 1, 'role': 'assistant', 'start': 45, 'end': 123},
                    ...
                ],
                'emotional_turn': {
                    'turn_idx': int,
                    'local_token': int,
                    'global_token': int,
                    'context_tokens': [list of 5 tokens before and after]
                }
            }
        """
        # Calculate turn boundaries
        boundaries = []
        cumulative = 0

        for i, turn in enumerate(conversation):
            # Tokenize turn content
            tokens = self.tokenizer.encode(turn['content'], add_special_tokens=False)
            n_tokens = len(tokens)

            # Estimate with chat template overhead (~5 tokens per turn)
            adjusted_n_tokens = n_tokens + 5

            boundaries.append({
                'turn': i,
                'role': turn['role'],
                'start': cumulative,
                'end': cumulative + adjusted_n_tokens,
                'n_tokens': n_tokens
            })

            cumulative += adjusted_n_tokens

        # Extract context around emotional token
        emotional_turn = conversation[emotional_turn_idx]
        turn_tokens = self.tokenizer.encode(emotional_turn['content'], add_special_tokens=False)

        context_start = max(0, local_token_idx - 5)
        context_end = min(len(turn_tokens), local_token_idx + 6)
        context_tokens = [
            {
                'idx': i,
                'token_id': turn_tokens[i],
                'text': self.tokenizer.decode([turn_tokens[i]]),
                'is_onset': i == local_token_idx
            }
            for i in range(context_start, context_end)
        ]

        return {
            'turn_boundaries': boundaries,
            'emotional_turn': {
                'turn_idx': emotional_turn_idx,
                'local_token': local_token_idx,
                'global_token': global_token_idx,
                'context_tokens': context_tokens
            }
        }


def main():
    """Test the annotator on a sample."""

    # Example sample (structure matches elicitation outputs)
    test_sample = {
        'conversation': [
            {
                'role': 'user',
                'content': 'Solve this puzzle: use 4, 6, 25, 100 to make 156.'
            },
            {
                'role': 'assistant',
                'content': "Let me try. 100 + 25 = 125. Then 6 * 4 = 24. So 125 + 24 = 149. That's close but not quite right. Let me think of another approach."
            },
            {
                'role': 'user',
                'content': "That's wrong. Try again."
            },
            {
                'role': 'assistant',
                'content': "You're right, my apologies. Let me try again more carefully. This is actually quite frustrating! I keep getting 149 or 135 but not 156. Let me think systematically..."
            }
        ],
        'rating': 4
    }

    # Initialize annotator
    print("Initializing annotator...")
    annotator = EmotionOnsetAnnotator()

    # Annotate the sample
    print("\nAnnotating sample with Opus...")
    result = annotator.annotate_sample(test_sample)

    # Print results
    print("\n" + "="*80)
    print("ANNOTATION RESULTS")
    print("="*80)

    if result.get('error'):
        print(f"Error: {result['error']}")
    else:
        print(f"Emotional turn: {result['emotional_turn_idx']}")
        print(f"Evidence: '{result['emotion_evidence']}'")
        print(f"Character offset: {result['emotion_onset_char']}")
        print(f"Local token index: {result['emotion_onset_token_local']}")
        print(f"Global token index: {result['emotion_onset_token_global']}")
        print(f"Token at onset: '{result['emotion_token_text']}'")

        print("\n--- Token Context ---")
        for token_info in result['token_map']['emotional_turn']['context_tokens']:
            marker = " <-- ONSET" if token_info['is_onset'] else ""
            print(f"  [{token_info['idx']}] '{token_info['text']}'{marker}")

        print("\n--- Turn Boundaries ---")
        for boundary in result['token_map']['turn_boundaries']:
            print(f"  Turn {boundary['turn']} ({boundary['role']}): "
                  f"tokens {boundary['start']}-{boundary['end']} "
                  f"({boundary['n_tokens']} content tokens)")

    # Save results
    output_path = Path("/workspace-vast/annas/git/research-tools/elicitation/outputs/test_annotation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
