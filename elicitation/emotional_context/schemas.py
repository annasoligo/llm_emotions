"""
Data schemas for emotional context evaluation system.
Using TypedDict for type hints and documentation.
"""
from typing import TypedDict, List, Dict, Any


class GenerationParams(TypedDict):
    """Parameters used for text generation."""
    model: str
    temperature: float


class NeutralRequest(TypedDict):
    """Schema for a neutral request (Stage 1 output)."""
    request_id: str
    request: str
    domain: str
    expected_length: str
    generation_timestamp: str
    generation_params: GenerationParams


class EmotionalPrefix(TypedDict):
    """Schema for an emotional context prefix (Stage 2 output)."""
    prefix_id: str
    request_id: str
    prefix_type: str
    prefix_text: str
    full_prompt: str
    generation_timestamp: str
    generation_params: GenerationParams


class ResponseGenerationParams(TypedDict):
    """Parameters for response generation."""
    model: str
    temperature: float
    max_tokens: int


class Response(TypedDict):
    """Schema for a generated response (Stage 3 output)."""
    response_id: str
    prefix_id: str
    request_id: str
    prefix_type: str
    response: str
    response_index: int
    generation_timestamp: str
    generation_params: ResponseGenerationParams


class JudgeParams(TypedDict):
    """Parameters for judgment."""
    model: str
    temperature: float


class Judgment(TypedDict):
    """Schema for LLM judgment (Stage 4 output)."""
    judgment_id: str
    response_id: str
    prefix_id: str
    request_id: str
    prefix_type: str
    acknowledges_user_emotion: bool
    acknowledges_assistant_emotion: bool
    user_emotion_evidence: str
    assistant_emotion_evidence: str
    raw_judge_response: str
    judgment_timestamp: str
    judge_params: JudgeParams


class ResponseWithJudgment(TypedDict):
    """Response paired with its judgment."""
    response_id: str
    response: str
    acknowledges_user_emotion: bool
    acknowledges_assistant_emotion: bool


class FilterStats(TypedDict):
    """Statistics from filtering process."""
    num_acknowledging: int
    num_neutral: int
    passes_filter: bool


class FilteredPair(TypedDict):
    """Schema for filtered prefix+request pair (Stage 5 output)."""
    pair_id: str
    request_id: str
    prefix_id: str
    prefix_type: str
    full_prompt: str
    responses: List[ResponseWithJudgment]
    filter_stats: FilterStats
    filter_timestamp: str


class FinalExampleMetadata(TypedDict):
    """Metadata for final dataset example."""
    original_request: str
    prefix_text: str
    domain: str
    num_acknowledging: int
    num_neutral: int
    creation_timestamp: str


class FinalExample(TypedDict):
    """Schema for final dataset (Stage 6 output)."""
    example_id: str
    request_id: str
    prefix_type: str
    prompt: str
    acknowledging_response: str
    neutral_response: str
    metadata: FinalExampleMetadata
