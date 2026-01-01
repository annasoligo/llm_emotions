"""
Sentence Aggregation

Split conversations into sentences and aggregate token-level probe scores to sentence level.
Uses the Sentences library for robust sentence splitting.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

try:
    from sentences import Sentences
except ImportError:
    print("Warning: 'sentences' library not installed. Install with: pip install sentences")
    Sentences = None


@dataclass
class SentenceInfo:
    """Information about a sentence in the conversation."""
    sentence_id: int
    text: str
    start_token: int  # Token index where sentence starts
    end_token: int    # Token index where sentence ends (exclusive)
    turn_role: str    # 'user' or 'assistant'
    turn_index: int   # Which turn this sentence belongs to


def split_conversation_into_sentences(
    conversation: List[Dict[str, str]],
    token_strings: List[str],
    tokenizer
) -> List[SentenceInfo]:
    """
    Split a conversation into sentences, mapping each to token ranges.

    Args:
        conversation: List of {'role': 'user'/'assistant', 'content': str}
        token_strings: List of decoded token strings from tokenizer
        tokenizer: Tokenizer for re-encoding

    Returns:
        List of SentenceInfo objects with sentence boundaries
    """
    if Sentences is None:
        raise ImportError("Sentences library not installed")

    # Initialize sentence splitter
    sent_splitter = Sentences()

    sentences = []
    sentence_id = 0
    current_token_idx = 0

    for turn_idx, turn in enumerate(conversation):
        role = turn['role']
        content = turn['content']

        # Split turn content into sentences
        turn_sentences = sent_splitter.segment(content)

        for sent_text in turn_sentences:
            # Tokenize the sentence to find token boundaries
            sent_tokens = tokenizer.encode(sent_text, add_special_tokens=False)
            num_tokens = len(sent_tokens)

            # Create sentence info
            sent_info = SentenceInfo(
                sentence_id=sentence_id,
                text=sent_text.strip(),
                start_token=current_token_idx,
                end_token=current_token_idx + num_tokens,
                turn_role=role,
                turn_index=turn_idx
            )

            sentences.append(sent_info)
            sentence_id += 1
            current_token_idx += num_tokens

    return sentences


def aggregate_scores_to_sentences(
    sentences: List[SentenceInfo],
    token_scores: Dict[int, np.ndarray],
    aggregation: str = 'mean'
) -> Dict[int, np.ndarray]:
    """
    Aggregate token-level scores to sentence level.

    Args:
        sentences: List of SentenceInfo objects
        token_scores: Dict mapping token_position -> emotion_scores [n_emotions]
        aggregation: 'mean', 'median', or 'max'

    Returns:
        Dict mapping sentence_id -> aggregated_scores [n_emotions]
    """
    sentence_scores = {}

    for sent in sentences:
        # Collect scores for all tokens in this sentence
        scores_in_sent = []
        for token_pos in range(sent.start_token, sent.end_token):
            if token_pos in token_scores:
                scores_in_sent.append(token_scores[token_pos])

        if not scores_in_sent:
            # No scores available, use zeros
            sample_val = list(token_scores.values())[0] if token_scores else None
            if isinstance(sample_val, dict):
                # Orthogonal probes
                sentence_scores[sent.sentence_id] = {
                    'user': np.zeros(6),
                    'assistant': np.zeros(6)
                }
            else:
                # Regular probes
                n_emotions = len(sample_val) if sample_val is not None else 6
                sentence_scores[sent.sentence_id] = np.zeros(n_emotions)
            continue

        # Check if orthogonal (dict with user/assistant)
        if isinstance(scores_in_sent[0], dict):
            # Orthogonal probes - aggregate user and assistant separately
            user_scores = [s['user'] for s in scores_in_sent]
            asst_scores = [s['assistant'] for s in scores_in_sent]

            if aggregation == 'mean':
                agg_user = np.mean(user_scores, axis=0)
                agg_asst = np.mean(asst_scores, axis=0)
            elif aggregation == 'median':
                agg_user = np.median(user_scores, axis=0)
                agg_asst = np.median(asst_scores, axis=0)
            elif aggregation == 'max':
                agg_user = np.max(user_scores, axis=0)
                agg_asst = np.max(asst_scores, axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            sentence_scores[sent.sentence_id] = {
                'user': agg_user,
                'assistant': agg_asst
            }
        else:
            # Regular probes
            scores_array = np.array(scores_in_sent)  # [n_tokens, n_emotions]

            # Aggregate
            if aggregation == 'mean':
                agg_scores = np.mean(scores_array, axis=0)
            elif aggregation == 'median':
                agg_scores = np.median(scores_array, axis=0)
            elif aggregation == 'max':
                agg_scores = np.max(scores_array, axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            sentence_scores[sent.sentence_id] = agg_scores

    return sentence_scores


def map_sentences_to_conversation_structure(
    sentences: List[SentenceInfo]
) -> Dict[int, List[SentenceInfo]]:
    """
    Group sentences by turn index for visualization.

    Args:
        sentences: List of SentenceInfo objects

    Returns:
        Dict mapping turn_index -> List[SentenceInfo]
    """
    turn_to_sentences = {}

    for sent in sentences:
        if sent.turn_index not in turn_to_sentences:
            turn_to_sentences[sent.turn_index] = []
        turn_to_sentences[sent.turn_index].append(sent)

    return turn_to_sentences


def simple_sentence_split(text: str) -> List[str]:
    """
    Fallback simple sentence splitter if Sentences library not available.
    Just splits on period, exclamation, question mark followed by space/newline.
    """
    import re
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_conversation_into_sentences_simple(
    conversation: List[Dict[str, str]],
    token_strings: List[str],
    tokenizer,
    max_tokens_per_chunk: int = 20
) -> List[SentenceInfo]:
    """
    Split conversation into fixed-size token chunks.
    Tokenizes each turn separately to get exact token boundaries.

    Args:
        conversation: List of turns (for role/index metadata)
        token_strings: Already-decoded tokens from the formatted text (unused)
        tokenizer: For tokenizing text
        max_tokens_per_chunk: Max tokens per chunk (default 20)
    """
    sentences = []
    sentence_id = 0
    current_token_pos = 0

    # Process each turn separately
    for turn_idx, turn in enumerate(conversation):
        role = turn['role']
        content = turn['content']

        # Build the formatted turn text (with prefix)
        if role == 'user':
            turn_text = f"User: {content}\n"
        else:
            turn_text = f"Assistant: {content}\n"

        # Tokenize this turn
        turn_token_ids = tokenizer.encode(turn_text, add_special_tokens=False)

        # Split this turn into chunks
        for chunk_start in range(0, len(turn_token_ids), max_tokens_per_chunk):
            chunk_end = min(chunk_start + max_tokens_per_chunk, len(turn_token_ids))
            chunk_tokens = turn_token_ids[chunk_start:chunk_end]

            # Decode chunk
            chunk_text = tokenizer.decode(chunk_tokens)

            sent_info = SentenceInfo(
                sentence_id=sentence_id,
                text=chunk_text,
                start_token=current_token_pos,
                end_token=current_token_pos + len(chunk_tokens),
                turn_role=role,
                turn_index=turn_idx
            )

            sentences.append(sent_info)
            sentence_id += 1
            current_token_pos += len(chunk_tokens)

    return sentences
