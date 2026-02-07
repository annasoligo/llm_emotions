"""OpenRouter inference backend for closed model evaluation.

Provides async batch generation via OpenRouter API with:
- Semaphore-based concurrency control
- Exponential backoff on rate limits (429) and server errors (5xx)
- OpenAI-compatible chat format (no model-specific tokens)
"""

import asyncio
import os
from typing import List, Dict, Any, Optional

import httpx


# Model shortcuts for common models
OPENROUTER_MODELS = {
    "gemini-flash": "google/gemini-2.0-flash",
    "gemini-flash-lite": "google/gemini-2.0-flash-lite",
    "gemini-pro": "google/gemini-pro-1.5",
    "gemini-pro-2": "google/gemini-2.0-pro",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "claude-haiku": "anthropic/claude-3-5-haiku",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default configuration
DEFAULT_MAX_CONCURRENT = 20
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 2.0


class OpenRouterInference:
    """Async inference via OpenRouter API.

    Example:
        >>> inference = OpenRouterInference(
        ...     model="google/gemini-2.0-flash",
        ...     temperature=0.7,
        ...     max_tokens=2048,
        ... )
        >>> responses = await inference.generate_batch(messages_list)
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        disable_thinking: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OpenRouter inference backend.

        Args:
            model: OpenRouter model ID (e.g., "google/gemini-2.0-flash")
                   or shortcut from OPENROUTER_MODELS
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            max_concurrent: Maximum concurrent API requests
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
            backoff_base: Base for exponential backoff
            disable_thinking: If True, disable thinking/reasoning for models that support it
            extra_params: Additional parameters to pass to the API
        """
        # Resolve model shortcut if needed
        self.model = OPENROUTER_MODELS.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.disable_thinking = disable_thinking
        self.extra_params = extra_params or {}

        # Get API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your key at https://openrouter.ai/keys"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_single(
        self,
        messages: List[Dict[str, str]],
        client: Optional[httpx.AsyncClient] = None,
    ) -> str:
        """Generate a single response with retry logic.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            client: Optional shared httpx client

        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add thinking/reasoning control for supported models
        if self.disable_thinking:
            # For Qwen models with thinking
            if "qwen" in self.model.lower():
                payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            # For OpenAI models - use reasoning_effort parameter
            if "openai" in self.model.lower() or "gpt" in self.model.lower():
                payload["reasoning"] = {"effort": "none"}
            # For Grok 4.1-fast (grok-4 requires reasoning, but 4.1-fast doesn't)
            if "grok-4.1" in self.model.lower():
                payload["reasoning"] = {"effort": "none"}
            # Note: Claude Sonnet doesn't have thinking by default
            # Note: Grok 4 (non-fast) requires reasoning and cannot disable it

        # Add any extra parameters
        payload.update(self.extra_params)

        should_close = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            should_close = True

        try:
            last_exception = None

            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        OPENROUTER_BASE_URL,
                        headers=self._get_headers(),
                        json=payload,
                    )

                    # Handle rate limits and server errors with retry
                    # 403 can sometimes be a temporary rate limit from providers
                    if response.status_code in [403, 429, 500, 502, 503, 504]:
                        if attempt < self.max_retries - 1:
                            wait_time = self.backoff_base ** attempt
                            print(f"  [Retrying after {response.status_code}, attempt {attempt + 1}]")
                            await asyncio.sleep(wait_time)
                            continue
                        response.raise_for_status()

                    response.raise_for_status()

                    # Handle JSON decode errors (can happen with incomplete responses)
                    try:
                        data = response.json()
                    except Exception as json_err:
                        if attempt < self.max_retries - 1:
                            wait_time = self.backoff_base ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        raise ValueError(f"Failed to parse JSON response after {self.max_retries} retries: {json_err}")

                    # Handle API-level errors
                    if "error" in data:
                        error_msg = data["error"].get("message", str(data["error"]))
                        raise ValueError(f"OpenRouter API error: {error_msg}")

                    return data["choices"][0]["message"]["content"]

                except httpx.HTTPStatusError as e:
                    last_exception = e
                    if attempt < self.max_retries - 1 and e.response.status_code in [403, 429, 500, 502, 503, 504]:
                        wait_time = self.backoff_base ** attempt
                        await asyncio.sleep(wait_time)
                    else:
                        raise

                except (httpx.RequestError, httpx.TimeoutException) as e:
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        wait_time = self.backoff_base ** attempt
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            raise last_exception

        finally:
            if should_close:
                await client.aclose()

    async def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        show_progress: bool = True,
    ) -> List[str]:
        """Generate responses for multiple conversations with concurrency control.

        Args:
            messages_list: List of message histories, each a list of
                          {"role": "user"|"assistant", "content": "..."}
            show_progress: Whether to print progress updates

        Returns:
            List of generated text responses (same order as input)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [None] * len(messages_list)
        completed = 0
        total = len(messages_list)

        async def generate_with_index(idx: int, messages: List[Dict[str, str]], client: httpx.AsyncClient) -> None:
            nonlocal completed
            async with semaphore:
                results[idx] = await self.generate_single(messages, client)
                completed += 1
                if show_progress and completed % 10 == 0:
                    print(f"  Progress: {completed}/{total} ({100*completed/total:.0f}%)")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                generate_with_index(i, msgs, client)
                for i, msgs in enumerate(messages_list)
            ]
            await asyncio.gather(*tasks)

        return results

    def generate_batch_sync(
        self,
        messages_list: List[List[Dict[str, str]]],
        show_progress: bool = True,
    ) -> List[str]:
        """Synchronous wrapper for generate_batch.

        Args:
            messages_list: List of message histories
            show_progress: Whether to print progress updates

        Returns:
            List of generated text responses
        """
        return asyncio.run(self.generate_batch(messages_list, show_progress))
