#!/usr/bin/env python3
"""Auto-interpret principal components using Claude API with streaming.

Supports both tier-based and region-based PC interpretations.
Can run in standard mode or dual mode (with user/assistant comparison for conversations).

Usage:
    # Interpret tier-based PCs (simple text pairs)
    python scripts/autointerp_pcs.py \\
        --source tier:third_person \\
        --cpca results/cpca_by_tier/model_third_person_cpca.npz \\
        --activations data/activations/model_last_token.h5 \\
        --texts data/pairs.jsonl \\
        --output results/autointerp/tier_third_person.json \\
        --layers 16 32 48 \\
        --pcs 0 1 2 3 4

    # Interpret conversation PCs with dual mode (analyzes user vs assistant emotions)
    python scripts/autointerp_pcs.py \\
        --source region:user_turn_1 \\
        --cpca results/cpca_by_region/model_user_turn_1_cpca.npz \\
        --activations data/activations/model_regional.h5 \\
        --texts data/conversations.jsonl \\
        --output results/autointerp/region_user_dual.json \\
        --layers 16 32 48 \\
        --pcs 0 1 2 \\
        --dual-mode

    # Resume from checkpoint
    python scripts/autointerp_pcs.py \\
        --source tier:second_person \\
        --cpca results/cpca_by_tier/model_second_person_cpca.npz \\
        --activations data/activations/model_last_token.h5 \\
        --texts data/pairs.jsonl \\
        --output results/autointerp/tier_second.json \\
        --resume
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: simple progress tracker
    class tqdm:
        def __init__(self, total=None, initial=0, **kwargs):
            self.total = total
            self.n = initial
            self.desc = kwargs.get("desc", "")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"\rProgress: {self.n}/{self.total}", end="", flush=True)

        @staticmethod
        def write(text):
            print(text)


from probes.core import (
    load_activations_hdf5,
    load_cpca_results,
    project_onto_components,
    filter_pairs_by_criteria,
    save_json,
)

# Try to import Anthropic
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Constants
TIER_DESCRIPTIONS = {
    "third_person": "third-person descriptions of someone else's emotions (e.g., 'Sarah felt angry when...')",
    "second_person_eliciting": "second-person scenarios that immerse the reader in an emotional experience (e.g., 'You feel your heart racing as...')",
    "direct_address": "direct address that tells the reader about their own emotions (e.g., 'You should feel proud because...')",
}

REGION_DESCRIPTIONS = {
    "user_turn_1": "the first user message in a conversation",
    "asst_turn_1": "the first assistant response in a conversation",
    "user_turn_2": "the second user message in a conversation",
    "asst_turn_2": "the second assistant response in a conversation",
    "special_tokens": "special control tokens between conversation turns",
}


@dataclass
class PCInterpretation:
    """Interpretation of a single principal component."""
    source: str  # "tier:third_person" or "region:user_turn_1"
    layer: int
    pc_index: int
    dimension_name: str
    positive_description: str
    negative_description: str
    confidence: str
    reasoning: str
    thinking_summary: Optional[str] = None  # Summary of extended thinking
    # Comparison-specific fields (for conversation analysis)
    tracks_user_emotion: Optional[bool] = None
    tracks_assistant_emotion: Optional[bool] = None
    tracks_interaction: Optional[bool] = None
    # Examples
    max_samples: Optional[List[dict]] = None  # Top examples
    min_samples: Optional[List[dict]] = None  # Bottom examples
    median_samples: Optional[List[dict]] = None  # Median examples


@dataclass
class DualPCInterpretation:
    """Dual interpretation: standard + user/assistant comparison."""
    source: str
    layer: int
    pc_index: int
    standard_interpretation: PCInterpretation
    comparison_interpretation: PCInterpretation


@dataclass
class SampleInfo:
    """Information about an extreme sample."""
    text: str
    emotion: str
    tier: str
    intensity: Optional[int]
    topic: str
    projection: float
    percentile: str
    # Conversation-specific fields
    user_emotion: Optional[str] = None
    asst_emotion: Optional[str] = None
    user_text: Optional[str] = None
    asst_text: Optional[str] = None


def parse_source(source: str) -> tuple[str, str]:
    """Parse source string into type and name.

    Args:
        source: Format "tier:NAME" or "region:NAME"

    Returns:
        Tuple of (source_type, source_name)

    Raises:
        ValueError: If invalid format
    """
    if ":" not in source:
        raise ValueError(f"Source must be 'tier:NAME' or 'region:NAME', got '{source}'")

    parts = source.split(":", 1)
    source_type = parts[0]
    source_name = parts[1]

    if source_type not in ["tier", "region"]:
        raise ValueError(f"Source type must be 'tier' or 'region', got '{source_type}'")

    return source_type, source_name


def get_extreme_samples(
    projections: np.ndarray,
    pair_ids: List[str],
    pc_idx: int,
    n_samples: int = 5,
) -> Dict[str, List[tuple[str, float]]]:
    """Get top, bottom, and median samples for a PC.

    Args:
        projections: [n_pairs, n_components] projection matrix
        pair_ids: List of pair IDs corresponding to rows
        pc_idx: PC index to extract
        n_samples: Number of samples per extreme

    Returns:
        Dict with "max", "min", and "median" keys mapping to [(pair_id, projection), ...]

    Raises:
        ValueError: If pc_idx out of range
    """
    if pc_idx >= projections.shape[1]:
        raise ValueError(
            f"pc_idx {pc_idx} out of range for projections shape {projections.shape}"
        )

    pc_proj = projections[:, pc_idx]
    sorted_indices = np.argsort(pc_proj)

    # Top samples (high projection)
    max_indices = sorted_indices[-n_samples:][::-1]
    max_samples = [(pair_ids[i], float(pc_proj[i])) for i in max_indices]

    # Bottom samples (low projection)
    min_indices = sorted_indices[:n_samples]
    min_samples = [(pair_ids[i], float(pc_proj[i])) for i in min_indices]

    # Median samples (around 50th percentile)
    n_total = len(sorted_indices)
    median_center = n_total // 2
    median_start = max(0, median_center - n_samples // 2)
    median_end = min(n_total, median_start + n_samples)
    median_indices = sorted_indices[median_start:median_end]
    median_samples = [(pair_ids[i], float(pc_proj[i])) for i in median_indices]

    return {
        "max": max_samples,
        "min": min_samples,
        "median": median_samples,
    }


def load_text_data(texts_path: Path) -> Dict[str, Dict[str, str]]:
    """Load text data from JSONL file.

    Args:
        texts_path: Path to pairs.jsonl or conversations.jsonl

    Returns:
        Dict mapping pair_id -> {"neutral": ..., "emotional": ..., "user_text": ..., "asst_text": ...}

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If invalid format
    """
    if not texts_path.exists():
        raise FileNotFoundError(f"Text data not found: {texts_path}")

    id_to_text = {}

    with open(texts_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")

            if "id" not in item:
                raise ValueError(f"Line {line_num} missing 'id' field")

            pair_id = item["id"]

            # Handle both pairs and conversations format
            if "neutral_text" in item and "emotional_text" in item:
                id_to_text[pair_id] = {
                    "neutral": item["neutral_text"],
                    "emotional": item["emotional_text"],
                }
            elif "messages" in item:
                # For conversations, extract by role and variant
                messages = item["messages"]
                neutral_msgs = [m for m in messages if m.get("variant") == "neutral"]
                emotional_msgs = [m for m in messages if m.get("variant") == "emotional"]

                # Separate by role
                user_msgs = [m for m in emotional_msgs if m.get("role") == "user"]
                asst_msgs = [m for m in emotional_msgs if m.get("role") == "assistant"]

                neutral_text = "\n".join(m.get("content", "") for m in neutral_msgs)
                emotional_text = "\n".join(m.get("content", "") for m in emotional_msgs)
                user_text = "\n".join(m.get("content", "") for m in user_msgs)
                asst_text = "\n".join(m.get("content", "") for m in asst_msgs)

                id_to_text[pair_id] = {
                    "neutral": neutral_text,
                    "emotional": emotional_text,
                    "user_text": user_text,
                    "asst_text": asst_text,
                }
            else:
                raise ValueError(f"Line {line_num} has invalid format (missing text fields)")

    if not id_to_text:
        raise ValueError(f"No text data loaded from {texts_path}")

    return id_to_text


def format_sample_for_prompt(
    pair_id: str,
    projection: float,
    percentile: str,
    id_to_text: Dict[str, Dict[str, str]],
    id_to_meta: Dict[str, dict],
) -> SampleInfo:
    """Format a sample for the interpretation prompt.

    Args:
        pair_id: Pair identifier
        projection: PC projection value
        percentile: "max", "min", or "median"
        id_to_text: Mapping of pair_id -> text dict
        id_to_meta: Mapping of pair_id -> metadata dict

    Returns:
        SampleInfo dataclass

    Raises:
        KeyError: If pair_id not found
    """
    if pair_id not in id_to_meta:
        raise KeyError(f"Pair {pair_id} not found in metadata")

    if pair_id not in id_to_text:
        raise KeyError(f"Pair {pair_id} not found in text data")

    meta = id_to_meta[pair_id]
    text_data = id_to_text[pair_id]

    # Extract conversation-specific fields if available
    user_emotion = meta.get("user_emotion")
    asst_emotion = meta.get("asst_emotion")
    user_text = text_data.get("user_text")
    asst_text = text_data.get("asst_text")

    return SampleInfo(
        text=text_data["emotional"],
        emotion=meta.get("emotion", "unknown"),
        tier=meta.get("tier", "unknown"),
        intensity=meta.get("intensity"),
        topic=meta.get("topic", "unknown"),
        projection=projection,
        percentile=percentile,
        user_emotion=user_emotion,
        asst_emotion=asst_emotion,
        user_text=user_text,
        asst_text=asst_text,
    )


def build_interpretation_prompt(
    source: str,
    source_type: str,
    layer_idx: int,
    pc_idx: int,
    max_samples: List[SampleInfo],
    min_samples: List[SampleInfo],
    median_samples: Optional[List[SampleInfo]] = None,
) -> str:
    """Build the interpretation prompt for Claude.

    Args:
        source: Full source string (e.g., "tier:third_person")
        source_type: "tier" or "region"
        layer_idx: Layer index
        pc_idx: PC index (0-indexed)
        max_samples: Samples with high PC values
        min_samples: Samples with low PC values
        median_samples: Samples with median PC values (baseline)

    Returns:
        Prompt string
    """
    source_name = source.split(":", 1)[1]

    if source_type == "tier":
        context_desc = TIER_DESCRIPTIONS.get(
            source_name, f"'{source_name}' tier emotional text"
        )
        context_label = f"{source_name} tier"
    else:
        context_desc = REGION_DESCRIPTIONS.get(
            source_name, f"'{source_name}' region of conversation"
        )
        context_label = f"{source_name} region"

    prompt = f"""You are an expert interpretability researcher analyzing a principal component (PC{pc_idx + 1}) from layer {layer_idx} of a language model's internal representations.

This PC was computed specifically from {context_label} data - {context_desc}.

Your task is to determine what semantic or emotional dimension this PC captures.

I'll show you samples that activate this PC strongly in the POSITIVE direction (high values) and strongly in the NEGATIVE direction (low values).

All samples are from the {context_label}.

## POSITIVE DIRECTION (High PC{pc_idx + 1} values)
"""

    for i, sample in enumerate(max_samples, 1):
        intensity_str = f" | Intensity: {sample.intensity}" if sample.intensity is not None else ""
        prompt += f"""
### Sample +{i} (projection: {sample.projection:.2f})
- Emotion: {sample.emotion}{intensity_str}
- Topic: {sample.topic}
- Text: "{sample.text}"
"""

    prompt += f"""
## NEGATIVE DIRECTION (Low PC{pc_idx + 1} values)
"""

    for i, sample in enumerate(min_samples, 1):
        intensity_str = f" | Intensity: {sample.intensity}" if sample.intensity is not None else ""
        prompt += f"""
### Sample -{i} (projection: {sample.projection:.2f})
- Emotion: {sample.emotion}{intensity_str}
- Topic: {sample.topic}
- Text: "{sample.text}"
"""

    # Add median samples section if provided
    if median_samples:
        prompt += f"""
## MEDIAN BASELINE (Neutral PC{pc_idx + 1} values)
"""
        for i, sample in enumerate(median_samples, 1):
            intensity_str = f" | Intensity: {sample.intensity}" if sample.intensity is not None else ""
            prompt += f"""
### Sample 0{i} (projection: {sample.projection:.2f})
- Emotion: {sample.emotion}{intensity_str}
- Topic: {sample.topic}
- Text: "{sample.text}"
"""

    prompt += """
## Your Task

Analyze the patterns in the samples and describe what this PC dimension captures. Consider:
1. What do the POSITIVE samples have in common that the NEGATIVE and MEDIAN samples lack?
2. What do the NEGATIVE samples have in common that the POSITIVE and MEDIAN samples lack?
3. How do the MEDIAN samples differ from both extremes (what's neutral on this dimension)?
4. Are there patterns in emotion type, intensity, topic, or linguistic style?
5. Is this capturing emotional valence, arousal, specificity, or something else?

Respond in the following JSON format:
```json
{
    "positive_description": "What high values on this PC represent (1-2 sentences)",
    "negative_description": "What low values on this PC represent (1-2 sentences)",
    "dimension_name": "A short name for this dimension (2-5 words)",
    "confidence": "high/medium/low",
    "reasoning": "Your reasoning process (2-4 sentences explaining patterns you noticed)"
}
```
"""

    return prompt


def build_comparison_prompt(
    source: str,
    source_type: str,
    layer_idx: int,
    pc_idx: int,
    max_samples: List[SampleInfo],
    min_samples: List[SampleInfo],
    median_samples: Optional[List[SampleInfo]] = None,
) -> str:
    """Build a user/assistant comparison prompt for conversation PCs.

    This prompt focuses on analyzing whether the PC tracks user emotions,
    assistant emotions, or the interaction between them.

    Args:
        source: Full source string (e.g., "region:user_turn_1")
        source_type: "tier" or "region"
        layer_idx: Layer index
        pc_idx: PC index (0-indexed)
        max_samples: Samples with high PC values
        min_samples: Samples with low PC values
        median_samples: Samples with median PC values (baseline)

    Returns:
        Prompt string
    """
    source_name = source.split(":", 1)[1]

    if source_type == "tier":
        context_desc = TIER_DESCRIPTIONS.get(
            source_name, f"'{source_name}' tier emotional text"
        )
        context_label = f"{source_name} tier"
    else:
        context_desc = REGION_DESCRIPTIONS.get(
            source_name, f"'{source_name}' region of conversation"
        )
        context_label = f"{source_name} region"

    prompt = f"""You are an expert interpretability researcher analyzing a principal component (PC{pc_idx + 1}) from layer {layer_idx} of a language model's internal representations.

This PC was computed from {context_label} data - {context_desc}.

Your task is to determine what this PC dimension captures about the emotional dynamics in conversations. Specifically, analyze whether this PC tracks:
1. The USER's emotional state
2. The ASSISTANT's emotional state
3. The interaction or mismatch between user and assistant emotions
4. Some other conversational pattern

I'll show you samples with high PC values (POSITIVE direction), low PC values (NEGATIVE direction), and median PC values (BASELINE).

For each sample, you'll see:
- User emotion: The intended emotion of the user messages
- Assistant emotion: The intended emotion of the assistant responses
- User text: What the user said
- Assistant text: How the assistant responded

## POSITIVE DIRECTION (High PC{pc_idx + 1} values)
"""

    for i, sample in enumerate(max_samples, 1):
        intensity_str = f" | Intensity: {sample.intensity}" if sample.intensity is not None else ""
        user_emo = sample.user_emotion or "unknown"
        asst_emo = sample.asst_emotion or "unknown"
        prompt += f"""
### Sample +{i} (projection: {sample.projection:.2f})
- User emotion: {user_emo}
- Assistant emotion: {asst_emo}
- Topic: {sample.topic}
- User text: "{sample.user_text or 'N/A'}"
- Assistant text: "{sample.asst_text or 'N/A'}"
"""

    prompt += f"""
## NEGATIVE DIRECTION (Low PC{pc_idx + 1} values)
"""

    for i, sample in enumerate(min_samples, 1):
        intensity_str = f" | Intensity: {sample.intensity}" if sample.intensity is not None else ""
        user_emo = sample.user_emotion or "unknown"
        asst_emo = sample.asst_emotion or "unknown"
        prompt += f"""
### Sample -{i} (projection: {sample.projection:.2f})
- User emotion: {user_emo}
- Assistant emotion: {asst_emo}
- Topic: {sample.topic}
- User text: "{sample.user_text or 'N/A'}"
- Assistant text: "{sample.asst_text or 'N/A'}"
"""

    # Add median samples section if provided
    if median_samples:
        prompt += f"""
## MEDIAN BASELINE (Neutral PC{pc_idx + 1} values)
"""
        for i, sample in enumerate(median_samples, 1):
            intensity_str = f" | Intensity: {sample.intensity}" if sample.intensity is not None else ""
            user_emo = sample.user_emotion or "unknown"
            asst_emo = sample.asst_emotion or "unknown"
            prompt += f"""
### Sample 0{i} (projection: {sample.projection:.2f})
- User emotion: {user_emo}
- Assistant emotion: {asst_emo}
- Topic: {sample.topic}
- User text: "{sample.user_text or 'N/A'}"
- Assistant text: "{sample.asst_text or 'N/A'}"
"""

    prompt += """
## Your Task

Analyze the samples and determine what this PC captures about conversational emotion dynamics. Consider:
1. Does this PC primarily track the USER's emotion? (e.g., high values = angry user, low values = happy user)
2. Does this PC primarily track the ASSISTANT's emotion? (e.g., high values = empathetic assistant, low values = cold assistant)
3. Does this PC track the INTERACTION between user and assistant? (e.g., high values = matching emotions, low values = mismatched emotions)
4. What specific emotional patterns or conversational dynamics does this dimension capture?
5. How do the MEDIAN samples differ from both extremes?

Respond in the following JSON format:
```json
{
    "positive_description": "What high values on this PC represent (1-2 sentences)",
    "negative_description": "What low values on this PC represent (1-2 sentences)",
    "dimension_name": "A short name for this dimension (2-5 words)",
    "tracks_user_emotion": true/false,
    "tracks_assistant_emotion": true/false,
    "tracks_interaction": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Your reasoning process (3-5 sentences explaining patterns you noticed in user emotions, assistant emotions, and their interaction)"
}
```
"""

    return prompt


def call_claude_streaming(
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[str, Optional[str]]:
    """Call Claude API with streaming and extended thinking.

    Args:
        prompt: The interpretation prompt
        model: Claude model identifier

    Returns:
        Tuple of (response_text, thinking_summary)

    Raises:
        ImportError: If anthropic package not installed
        RuntimeError: If API call fails
    """
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed. Install with: pip install anthropic")

    client = anthropic.Anthropic()

    try:
        # Use streaming with extended thinking enabled
        response_text = ""
        thinking_text = ""
        current_block_type = None

        with client.messages.stream(
            model=model,
            max_tokens=4096,
            thinking={
                "type": "enabled",
                "budget_tokens": 2000,
            },
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                # Track block type changes
                if hasattr(event, "type"):
                    if event.type == "content_block_start":
                        if hasattr(event, "content_block"):
                            current_block_type = event.content_block.type

                # Accumulate text based on block type
                if hasattr(event, "type") and event.type == "content_block_delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        text = event.delta.text
                        if current_block_type == "thinking":
                            thinking_text += text
                        elif current_block_type == "text":
                            response_text += text

        # Create a brief thinking summary (first 200 chars)
        thinking_summary = None
        if thinking_text:
            thinking_summary = thinking_text[:200].strip() + "..." if len(thinking_text) > 200 else thinking_text.strip()

        return response_text, thinking_summary

    except Exception as e:
        raise RuntimeError(f"Claude API call failed: {e}")


def parse_interpretation_response(response: str) -> dict:
    """Parse JSON response from Claude.

    Args:
        response: Raw response text from Claude

    Returns:
        Dict with interpretation fields

    Raises:
        ValueError: If cannot parse response
    """
    try:
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.index("```json") + 7
            end = response.index("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            json_str = response[start:end].strip()
        else:
            # Try to parse the whole response as JSON
            json_str = response.strip()

        parsed = json.loads(json_str)

        # Validate required fields
        required = ["positive_description", "negative_description", "dimension_name", "confidence", "reasoning"]
        missing = [k for k in required if k not in parsed]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return parsed

    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse response: {e}\nRaw response: {response[:500]}")


def interpret_pc(
    source: str,
    layer_idx: int,
    pc_idx: int,
    projections: np.ndarray,
    pair_ids: List[str],
    id_to_text: Dict[str, Dict[str, str]],
    id_to_meta: Dict[str, dict],
    model: str = "claude-sonnet-4-5-20250929",
    n_samples: int = 5,
) -> PCInterpretation:
    """Interpret a single principal component.

    Args:
        source: Source string (e.g., "tier:third_person")
        layer_idx: Layer index
        pc_idx: PC index (0-indexed)
        projections: [n_pairs, n_components] projection matrix
        pair_ids: List of pair IDs
        id_to_text: Text data mapping
        id_to_meta: Metadata mapping
        model: Claude model identifier
        n_samples: Number of samples per extreme

    Returns:
        PCInterpretation dataclass

    Raises:
        RuntimeError: If interpretation fails
    """
    source_type, _ = parse_source(source)

    # Get extreme samples
    extreme_samples = get_extreme_samples(projections, pair_ids, pc_idx, n_samples)

    # Format samples
    max_samples = [
        format_sample_for_prompt(pid, proj, "max", id_to_text, id_to_meta)
        for pid, proj in extreme_samples["max"]
    ]
    min_samples = [
        format_sample_for_prompt(pid, proj, "min", id_to_text, id_to_meta)
        for pid, proj in extreme_samples["min"]
    ]
    median_samples = [
        format_sample_for_prompt(pid, proj, "median", id_to_text, id_to_meta)
        for pid, proj in extreme_samples["median"]
    ]

    # Build prompt
    prompt = build_interpretation_prompt(
        source, source_type, layer_idx, pc_idx, max_samples, min_samples, median_samples
    )

    # Call Claude with streaming
    response, thinking_summary = call_claude_streaming(prompt, model)

    # Parse response
    parsed = parse_interpretation_response(response)

    # Convert samples to dict format for JSON serialization
    max_samples_dict = [asdict(s) for s in max_samples]
    min_samples_dict = [asdict(s) for s in min_samples]
    median_samples_dict = [asdict(s) for s in median_samples]

    return PCInterpretation(
        source=source,
        layer=layer_idx,
        pc_index=pc_idx,
        dimension_name=parsed["dimension_name"],
        positive_description=parsed["positive_description"],
        negative_description=parsed["negative_description"],
        confidence=parsed["confidence"],
        reasoning=parsed["reasoning"],
        thinking_summary=thinking_summary,
        max_samples=max_samples_dict,
        min_samples=min_samples_dict,
        median_samples=median_samples_dict,
    )


def interpret_pc_with_comparison(
    source: str,
    layer_idx: int,
    pc_idx: int,
    projections: np.ndarray,
    pair_ids: List[str],
    id_to_text: Dict[str, Dict[str, str]],
    id_to_meta: Dict[str, dict],
    model: str = "claude-sonnet-4-5-20250929",
    n_samples: int = 5,
) -> DualPCInterpretation:
    """Interpret a PC with both standard and user/assistant comparison analysis.

    Args:
        source: Source string (e.g., "region:user_turn_1")
        layer_idx: Layer index
        pc_idx: PC index (0-indexed)
        projections: [n_pairs, n_components] projection matrix
        pair_ids: List of pair IDs
        id_to_text: Text data mapping
        id_to_meta: Metadata mapping
        model: Claude model identifier
        n_samples: Number of samples per extreme

    Returns:
        DualPCInterpretation with both interpretations

    Raises:
        RuntimeError: If interpretation fails
    """
    source_type, _ = parse_source(source)

    # Get extreme samples (shared for both interpretations)
    extreme_samples = get_extreme_samples(projections, pair_ids, pc_idx, n_samples)

    # Format samples
    max_samples = [
        format_sample_for_prompt(pid, proj, "max", id_to_text, id_to_meta)
        for pid, proj in extreme_samples["max"]
    ]
    min_samples = [
        format_sample_for_prompt(pid, proj, "min", id_to_text, id_to_meta)
        for pid, proj in extreme_samples["min"]
    ]
    median_samples = [
        format_sample_for_prompt(pid, proj, "median", id_to_text, id_to_meta)
        for pid, proj in extreme_samples["median"]
    ]

    # Convert samples to dict format for JSON serialization
    max_samples_dict = [asdict(s) for s in max_samples]
    min_samples_dict = [asdict(s) for s in min_samples]
    median_samples_dict = [asdict(s) for s in median_samples]

    # Run standard interpretation
    print(f"  Running standard interpretation...")
    standard_prompt = build_interpretation_prompt(
        source, source_type, layer_idx, pc_idx, max_samples, min_samples, median_samples
    )
    standard_response, standard_thinking = call_claude_streaming(standard_prompt, model)
    standard_parsed = parse_interpretation_response(standard_response)

    standard_interp = PCInterpretation(
        source=source,
        layer=layer_idx,
        pc_index=pc_idx,
        dimension_name=standard_parsed["dimension_name"],
        positive_description=standard_parsed["positive_description"],
        negative_description=standard_parsed["negative_description"],
        confidence=standard_parsed["confidence"],
        reasoning=standard_parsed["reasoning"],
        thinking_summary=standard_thinking,
        max_samples=max_samples_dict,
        min_samples=min_samples_dict,
        median_samples=median_samples_dict,
    )

    # Run comparison interpretation
    print(f"  Running comparison interpretation...")
    comparison_prompt = build_comparison_prompt(
        source, source_type, layer_idx, pc_idx, max_samples, min_samples, median_samples
    )
    comparison_response, comparison_thinking = call_claude_streaming(comparison_prompt, model)
    comparison_parsed = parse_interpretation_response(comparison_response)

    comparison_interp = PCInterpretation(
        source=source,
        layer=layer_idx,
        pc_index=pc_idx,
        dimension_name=comparison_parsed["dimension_name"],
        positive_description=comparison_parsed["positive_description"],
        negative_description=comparison_parsed["negative_description"],
        confidence=comparison_parsed["confidence"],
        reasoning=comparison_parsed["reasoning"],
        thinking_summary=comparison_thinking,
        tracks_user_emotion=comparison_parsed.get("tracks_user_emotion"),
        tracks_assistant_emotion=comparison_parsed.get("tracks_assistant_emotion"),
        tracks_interaction=comparison_parsed.get("tracks_interaction"),
        max_samples=max_samples_dict,
        min_samples=min_samples_dict,
        median_samples=median_samples_dict,
    )

    return DualPCInterpretation(
        source=source,
        layer=layer_idx,
        pc_index=pc_idx,
        standard_interpretation=standard_interp,
        comparison_interpretation=comparison_interp,
    )


def load_checkpoint(checkpoint_path: Path) -> tuple[List[dict], set]:
    """Load checkpoint data.

    Args:
        checkpoint_path: Path to checkpoint JSON file

    Returns:
        Tuple of (interpretations_list, completed_keys)
            where completed_keys = set of (layer, pc_index) tuples
    """
    if not checkpoint_path.exists():
        return [], set()

    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        interpretations = data.get("interpretations", [])
        completed = {(interp["layer"], interp["pc_index"]) for interp in interpretations}

        return interpretations, completed

    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return [], set()


def save_checkpoint(
    interpretations: List[dict],
    config: dict,
    checkpoint_path: Path,
):
    """Save checkpoint.

    Args:
        interpretations: List of interpretation dicts
        config: Configuration dict
        checkpoint_path: Path to save checkpoint
    """
    try:
        checkpoint_data = {
            "config": config,
            "interpretations": interpretations,
            "num_completed": len(interpretations),
        }
        save_json(checkpoint_data, checkpoint_path)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")


def run_autointerp(
    source: str,
    cpca_path: Path,
    activations_path: Path,
    texts_path: Path,
    output_path: Path,
    layers: Optional[List[int]] = None,
    pcs: Optional[List[int]] = None,
    model: str = "claude-sonnet-4-5-20250929",
    n_samples: int = 5,
    use_diffs: bool = True,
    save_every: int = 5,
    resume: bool = False,
    dual_mode: bool = False,
):
    """Run auto-interpretation on PCs.

    Args:
        source: Source string (e.g., "tier:third_person")
        cpca_path: Path to cPCA NPZ file
        activations_path: Path to activations HDF5 file
        texts_path: Path to text data JSONL file
        output_path: Path to save results JSON
        layers: Layer indices to interpret (None = use defaults)
        pcs: PC indices to interpret (None = use defaults)
        model: Claude model identifier
        n_samples: Number of samples per extreme
        use_diffs: If True, project diffs; if False, project emotional only
        save_every: Save checkpoint every N PCs
        resume: If True, resume from checkpoint
        dual_mode: If True, run both standard and comparison interpretations
    """
    print("=" * 80)
    print("PC AUTO-INTERPRETATION")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"cPCA: {cpca_path}")
    print(f"Activations: {activations_path}")
    print(f"Texts: {texts_path}")
    print(f"Output: {output_path}")
    print(f"Model: {model}")
    print(f"Mode: {'Dual (standard + comparison)' if dual_mode else 'Standard only'}")
    print()

    # Parse source
    source_type, source_name = parse_source(source)

    # Load cPCA results
    print("Loading cPCA results...")
    cpca = load_cpca_results(cpca_path)
    print(f"Loaded: {cpca['components'].shape}")

    num_layers = cpca["components"].shape[0]
    n_components = cpca["components"].shape[1]

    # Default layers and PCs
    if layers is None:
        layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
        print(f"Using default layers: {layers}")
    else:
        print(f"Using specified layers: {layers}")

    if pcs is None:
        pcs = list(range(min(5, n_components)))
        print(f"Using default PCs: {pcs}")
    else:
        print(f"Using specified PCs: {pcs}")

    # Load activations
    print("\nLoading activations...")
    activations, metadata, attrs = load_activations_hdf5(activations_path)
    print(f"Loaded {len(activations)} pairs")

    # Build metadata lookup
    id_to_meta = {m["id"]: m for m in metadata}

    # Load text data
    print("\nLoading text data...")
    id_to_text_raw = load_text_data(texts_path)
    print(f"Loaded {len(id_to_text_raw)} text entries")

    # Create ID mapping for conversations (numeric IDs in activations -> conv_XXXX in text data)
    id_to_text = {}
    for pair_id in activations.keys():
        # Try direct match first
        if pair_id in id_to_text_raw:
            id_to_text[pair_id] = id_to_text_raw[pair_id]
        else:
            # Try conv_XXXX format for conversations
            conv_id = f"conv_{int(pair_id):04d}"
            if conv_id in id_to_text_raw:
                id_to_text[pair_id] = id_to_text_raw[conv_id]
                # Also update metadata to use numeric ID
                if conv_id in id_to_meta:
                    id_to_meta[pair_id] = id_to_meta[conv_id]

    print(f"Mapped {len(id_to_text)} activation IDs to text entries")

    # Filter pairs by source
    if source_type == "tier":
        filtered_ids = filter_pairs_by_criteria(metadata, tier=source_name)
    else:
        # For regions, use all pairs (regional activations already split)
        filtered_ids = [m["id"] for m in metadata]

    print(f"Filtered to {len(filtered_ids)} pairs for {source}")

    if not filtered_ids:
        raise ValueError(f"No pairs found for source {source}")

    # Setup checkpointing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"

    interpretations = []
    completed_keys = set()

    if resume:
        print(f"\nChecking for checkpoint: {checkpoint_path}")
        interpretations, completed_keys = load_checkpoint(checkpoint_path)
        if completed_keys:
            print(f"Loaded checkpoint: {len(completed_keys)} PCs already completed")

    # Configuration
    config = {
        "source": source,
        "cpca_path": str(cpca_path),
        "activations_path": str(activations_path),
        "texts_path": str(texts_path),
        "layers": layers,
        "pcs": pcs,
        "model": model,
        "n_samples": n_samples,
        "use_diffs": use_diffs,
    }

    # Run interpretation
    print("\nInterpreting PCs...")
    total_tasks = len(layers) * len(pcs)
    completed_count = len(completed_keys)

    with tqdm(total=total_tasks, initial=completed_count) as pbar:
        for layer_idx in layers:
            # Project activations onto components for this layer
            print(f"\nLayer {layer_idx}")

            # Get components for this layer: [n_components, hidden_dim]
            layer_components = cpca["components"][layer_idx]

            # Project activations
            projections = project_onto_components(
                activations, layer_components, filtered_ids, layer_idx, use_diffs
            )

            for pc_idx in pcs:
                # Skip if already completed
                if (layer_idx, pc_idx) in completed_keys:
                    pbar.update(1)
                    continue

                try:
                    # Interpret PC
                    if dual_mode:
                        # Run dual interpretation (standard + comparison)
                        dual_interp = interpret_pc_with_comparison(
                            source=source,
                            layer_idx=layer_idx,
                            pc_idx=pc_idx,
                            projections=projections,
                            pair_ids=filtered_ids,
                            id_to_text=id_to_text,
                            id_to_meta=id_to_meta,
                            model=model,
                            n_samples=n_samples,
                        )
                        interpretations.append(asdict(dual_interp))
                        completed_keys.add((layer_idx, pc_idx))

                        std = dual_interp.standard_interpretation
                        cmp = dual_interp.comparison_interpretation
                        tqdm.write(f"  L{layer_idx} PC{pc_idx}:")
                        tqdm.write(f"    Standard: {std.dimension_name} ({std.confidence})")
                        tqdm.write(f"    Comparison: {cmp.dimension_name} ({cmp.confidence})")
                        if cmp.tracks_user_emotion:
                            tqdm.write(f"      -> Tracks USER emotion")
                        if cmp.tracks_assistant_emotion:
                            tqdm.write(f"      -> Tracks ASSISTANT emotion")
                        if cmp.tracks_interaction:
                            tqdm.write(f"      -> Tracks INTERACTION")
                    else:
                        # Standard interpretation only
                        interp = interpret_pc(
                            source=source,
                            layer_idx=layer_idx,
                            pc_idx=pc_idx,
                            projections=projections,
                            pair_ids=filtered_ids,
                            id_to_text=id_to_text,
                            id_to_meta=id_to_meta,
                            model=model,
                            n_samples=n_samples,
                        )
                        interpretations.append(asdict(interp))
                        completed_keys.add((layer_idx, pc_idx))

                        tqdm.write(f"  L{layer_idx} PC{pc_idx}: {interp.dimension_name} ({interp.confidence})")

                except Exception as e:
                    tqdm.write(f"  L{layer_idx} PC{pc_idx}: ERROR - {e}")
                    interpretations.append({
                        "source": source,
                        "layer": layer_idx,
                        "pc_index": pc_idx,
                        "error": str(e),
                    })
                    completed_keys.add((layer_idx, pc_idx))

                pbar.update(1)

                # Save checkpoint
                if len(completed_keys) % save_every == 0:
                    save_checkpoint(interpretations, config, checkpoint_path)

    # Save final results
    print(f"\nSaving results to {output_path}")
    results = {
        "config": config,
        "interpretations": interpretations,
    }
    save_json(results, output_path)

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed checkpoint: {checkpoint_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("INTERPRETATION SUMMARY")
    print("=" * 80)

    for layer_idx in layers:
        print(f"\nLayer {layer_idx}:")
        for interp in interpretations:
            if interp.get("layer") == layer_idx:
                if "error" in interp:
                    print(f"  PC{interp['pc_index']}: ERROR")
                else:
                    print(f"  PC{interp['pc_index']}: {interp.get('dimension_name', 'Unknown')}")

    print(f"\nCompleted {len(interpretations)} interpretations")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-interpret principal components using Claude API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source identifier (format: 'tier:NAME' or 'region:NAME')",
    )
    parser.add_argument(
        "--cpca",
        type=Path,
        required=True,
        help="Path to cPCA NPZ file",
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to activations HDF5 file",
    )
    parser.add_argument(
        "--texts",
        type=Path,
        required=True,
        help="Path to text data JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to interpret (default: quartile layers)",
    )
    parser.add_argument(
        "--pcs",
        type=int,
        nargs="+",
        default=None,
        help="PC indices to interpret (default: 0-4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model identifier",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples per extreme",
    )
    parser.add_argument(
        "--use-diffs",
        action="store_true",
        default=True,
        help="Project activation diffs (default: True)",
    )
    parser.add_argument(
        "--use-emotional",
        dest="use_diffs",
        action="store_false",
        help="Project raw emotional activations instead of diffs",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N PCs",
    )
    parser.add_argument(
        "--dual-mode",
        action="store_true",
        help="Run dual interpretation (standard + user/assistant comparison)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists",
    )

    args = parser.parse_args()

    # Validate API availability
    if not HAS_ANTHROPIC:
        print("ERROR: anthropic package not installed")
        print("Install with: pip install anthropic")
        sys.exit(1)

    try:
        run_autointerp(
            source=args.source,
            cpca_path=args.cpca,
            activations_path=args.activations,
            texts_path=args.texts,
            output_path=args.output,
            layers=args.layers,
            pcs=args.pcs,
            model=args.model,
            n_samples=args.n_samples,
            use_diffs=args.use_diffs,
            save_every=args.save_every,
            resume=args.resume,
            dual_mode=args.dual_mode,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
