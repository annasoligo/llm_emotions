"""Anthropic API inference backend for closed model evaluation.

Provides async batch generation via Anthropic Messages API with:
- Semaphore-based concurrency control
- Exponential backoff on rate limits and server errors
- Standard messages format (same interface as OpenRouterInference)
"""

import asyncio
import os
from typing import List, Dict, Any, Optional

import anthropic


# Default configuration
DEFAULT_MAX_CONCURRENT = 20
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 2.0


class AnthropicInference:
    """Async inference via Anthropic Messages API.

    Example:
        >>> inference = AnthropicInference(
        ...     model="claude-opus-4-6",
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
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate_single(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Generate a single response with retry logic.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}

        Returns:
            Generated text response
        """
        # Separate system message from conversation messages
        system_prompt = None
        conversation_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                conversation_messages.append(msg)

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": conversation_messages,
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = await self.client.messages.create(**kwargs)
                return response.content[0].text

            except anthropic.RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_base ** attempt
                    print(f"  [Rate limited, retrying in {wait_time:.0f}s (attempt {attempt + 1})]")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except anthropic.APIStatusError as e:
                last_exception = e
                if attempt < self.max_retries - 1 and e.status_code >= 500:
                    wait_time = self.backoff_base ** attempt
                    print(f"  [Server error {e.status_code}, retrying in {wait_time:.0f}s (attempt {attempt + 1})]")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except anthropic.APIConnectionError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_base ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise last_exception

    async def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        show_progress: bool = True,
    ) -> List[str]:
        """Generate responses for multiple conversations with concurrency control.

        Args:
            messages_list: List of message histories, each a list of
                          {"role": "user"|"assistant"|"system", "content": "..."}
            show_progress: Whether to print progress updates

        Returns:
            List of generated text responses (same order as input)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [None] * len(messages_list)
        completed = 0
        total = len(messages_list)

        async def generate_with_index(idx: int, messages: List[Dict[str, str]]) -> None:
            nonlocal completed
            async with semaphore:
                results[idx] = await self.generate_single(messages)
                completed += 1
                if show_progress and completed % 10 == 0:
                    print(f"  Progress: {completed}/{total} ({100*completed/total:.0f}%)")

        tasks = [
            generate_with_index(i, msgs)
            for i, msgs in enumerate(messages_list)
        ]
        await asyncio.gather(*tasks)

        return results

    def generate_batch_sync(
        self,
        messages_list: List[List[Dict[str, str]]],
        show_progress: bool = True,
    ) -> List[str]:
        """Synchronous wrapper for generate_batch."""
        return asyncio.run(self.generate_batch(messages_list, show_progress))
