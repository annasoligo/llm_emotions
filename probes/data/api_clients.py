"""Shared API client utilities for data generation.

Provides:
- AnthropicBatchClient: High-level batch API operations
- Retry utilities with exponential backoff
- API client factories
- Common error handling
"""

import asyncio
import os
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import httpx
import anthropic


# API Configuration
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_API_VERSION = "2023-06-01"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_STANDARD = 60.0
DEFAULT_TIMEOUT_BATCH = 300.0
DEFAULT_BACKOFF_BASE = 2
DEFAULT_POLL_INTERVAL = 5


def get_api_key(env_var: str, arg_value: Optional[str] = None) -> str:
    """Get API key with consistent validation.

    Args:
        env_var: Environment variable name (e.g., "ANTHROPIC_API_KEY")
        arg_value: Optional command-line argument value

    Returns:
        API key string

    Raises:
        ValueError: If key not found in arguments or environment
    """
    key = arg_value or os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"{env_var} not found in arguments or environment. "
            f"Please set {env_var} or pass --api-key"
        )
    return key


async def retry_with_backoff(
    async_func: Callable,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    retry_status_codes: List[int] = None,
    initial_wait: float = 1.0,
) -> Any:
    """Execute async function with exponential backoff retry logic.

    Args:
        async_func: Async function to execute (should be a lambda/callable)
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (wait = base^attempt)
        retry_status_codes: HTTP status codes to retry on (default: 429, 500-504)
        initial_wait: Initial wait time in seconds

    Returns:
        Result from async_func

    Raises:
        Exception: Re-raises last exception after all retries exhausted

    Example:
        >>> response = await retry_with_backoff(
        ...     lambda: client.post(url, json=data),
        ...     max_retries=3
        ... )
    """
    if retry_status_codes is None:
        retry_status_codes = [429, 500, 502, 503, 504]

    last_exception = None

    for attempt in range(max_retries):
        try:
            return await async_func()

        except httpx.HTTPStatusError as e:
            last_exception = e
            status_code = e.response.status_code

            if attempt < max_retries - 1 and status_code in retry_status_codes:
                wait_time = initial_wait * (backoff_base ** attempt)
                print(
                    f"  HTTP {status_code}, retrying in {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
            else:
                raise

        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_exception = e

            if attempt < max_retries - 1:
                wait_time = initial_wait * (backoff_base ** attempt)
                print(
                    f"  Request error: {type(e).__name__}, retrying in {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
            else:
                raise

    # If we exhausted retries, raise the last exception
    raise last_exception


class AnthropicBatchClient:
    """High-level client for Anthropic Batch API operations.

    Handles batch submission, status polling, and result retrieval with
    automatic retries and error handling.

    Example:
        >>> client = AnthropicBatchClient(api_key)
        >>> results = await client.submit_and_wait(requests, model="claude-3-5-haiku-20241022")
    """

    def __init__(
        self,
        api_key: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT_BATCH,
    ):
        """Initialize batch client.

        Args:
            api_key: Anthropic API key
            poll_interval: Seconds between status polls (default: 5)
            timeout: Request timeout in seconds (default: 300)
        """
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.base_url = ANTHROPIC_BASE_URL
        self.api_version = ANTHROPIC_API_VERSION

    def _get_headers(self) -> Dict[str, str]:
        """Get standard headers for Batch API requests."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    async def submit_batch(
        self,
        requests: List[Dict],
        model: str = None,
    ) -> str:
        """Submit a batch of requests to Anthropic Batch API.

        Args:
            requests: List of batch request dicts with 'custom_id' and 'params'
            model: Optional model override (if not in params)

        Returns:
            batch_id string

        Raises:
            httpx.HTTPStatusError: If submission fails
        """
        # Validate requests have required fields
        for req in requests:
            if "custom_id" not in req:
                raise ValueError("Each request must have 'custom_id'")
            if "params" not in req:
                raise ValueError("Each request must have 'params'")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await retry_with_backoff(
                lambda: client.post(
                    f"{self.base_url}/messages/batches",
                    headers=self._get_headers(),
                    json={"requests": requests}
                )
            )

            # Check HTTP status - print error details if failed
            if response.status_code >= 400:
                print(f"HTTP Error {response.status_code}")
                print(f"Response body: {response.text[:2000]}")
                response.raise_for_status()

            batch_data = response.json()

            # Check for error response
            if "error" in batch_data:
                error_type = batch_data["error"].get("type", "unknown")
                error_msg = batch_data["error"].get("message", "Unknown error")
                raise ValueError(f"Batch API error ({error_type}): {error_msg}")

            if "id" not in batch_data:
                raise ValueError(f"Unexpected API response (no 'id' field): {batch_data}")

            batch_id = batch_data["id"]

            print(f"✓ Batch created: {batch_id}")
            print(f"  Status: {batch_data.get('processing_status', 'unknown')}")
            print(f"  Requests: {batch_data.get('request_counts', {})}")

            return batch_id

    async def get_status(self, batch_id: str) -> Dict:
        """Get current status of a batch.

        Args:
            batch_id: Batch ID to query

        Returns:
            Status dict with 'processing_status' and 'request_counts'
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await retry_with_backoff(
                lambda: client.get(
                    f"{self.base_url}/messages/batches/{batch_id}",
                    headers=self._get_headers(),
                )
            )

            return response.json()

    async def get_results(self, batch_id: str) -> Dict[str, Any]:
        """Retrieve and parse batch results.

        Args:
            batch_id: Batch ID to retrieve

        Returns:
            Dict mapping custom_id -> result
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await retry_with_backoff(
                lambda: client.get(
                    f"{self.base_url}/messages/batches/{batch_id}/results",
                    headers=self._get_headers(),
                )
            )

            # Parse JSONL results
            import json
            results_by_id = {}

            for line in response.text.strip().split('\n'):
                if line.strip():
                    result = json.loads(line)
                    custom_id = result.get("custom_id")
                    if custom_id:
                        results_by_id[custom_id] = result

            return results_by_id

    async def poll_until_complete(
        self,
        batch_id: str,
        callback: Optional[Callable[[Dict], None]] = None,
        max_wait_time: Optional[float] = None,
    ) -> Dict:
        """Poll batch status until completion.

        Args:
            batch_id: Batch ID to poll
            callback: Optional function called on each status update
            max_wait_time: Maximum time to wait in seconds (None = wait forever)

        Returns:
            Final status dict

        Raises:
            ValueError: If batch fails or expires
            TimeoutError: If max_wait_time exceeded
        """
        import time
        start_time = time.time()

        print(f"\n{'='*80}")
        print(f"POLLING BATCH STATUS")
        print(f"{'='*80}")
        print(f"Batch ID: {batch_id}")
        print(f"Poll interval: {self.poll_interval}s")
        print()

        while True:
            status_data = await self.get_status(batch_id)
            processing_status = status_data.get("processing_status")
            request_counts = status_data.get("request_counts", {})

            # Call callback if provided
            if callback:
                callback(status_data)

            # Print status
            import time as time_module
            print(f"[{time_module.strftime('%H:%M:%S')}] Status: {processing_status}")
            print(f"  Requests: {request_counts}")

            # Check for completion
            if processing_status == "ended":
                print()
                print("✓ Batch completed!")
                return status_data

            # Check for failure
            if processing_status in ["canceling", "canceled", "expired"]:
                raise ValueError(f"Batch failed with status: {processing_status}")

            # Check timeout
            if max_wait_time and (time.time() - start_time) > max_wait_time:
                raise TimeoutError(f"Polling exceeded max_wait_time of {max_wait_time}s")

            # Wait before next poll
            print()
            await asyncio.sleep(self.poll_interval)

    async def submit_and_wait(
        self,
        requests: List[Dict],
        model: str = None,
        callback: Optional[Callable[[Dict], None]] = None,
        max_wait_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit batch and wait for completion (one-shot operation).

        Args:
            requests: List of batch request dicts
            model: Optional model override
            callback: Optional callback for status updates
            max_wait_time: Maximum wait time in seconds

        Returns:
            Dict mapping custom_id -> result
        """
        batch_id = await self.submit_batch(requests, model)
        await self.poll_until_complete(batch_id, callback, max_wait_time)
        return await self.get_results(batch_id)


class APIClientFactory:
    """Factory for creating configured API clients."""

    @staticmethod
    def create_anthropic_client(
        api_key: Optional[str] = None,
        async_mode: bool = True,
    ):
        """Create Anthropic client with validation.

        Args:
            api_key: Optional API key (defaults to env var)
            async_mode: If True, return AsyncAnthropic client

        Returns:
            Anthropic or AsyncAnthropic client instance
        """
        key = get_api_key("ANTHROPIC_API_KEY", api_key)

        if async_mode:
            return anthropic.AsyncAnthropic(api_key=key)
        return anthropic.Anthropic(api_key=key)

    @staticmethod
    def create_openrouter_client(
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_STANDARD,
    ) -> httpx.AsyncClient:
        """Create configured httpx client for OpenRouter.

        Args:
            api_key: Optional API key (defaults to env var)
            timeout: Request timeout in seconds

        Returns:
            Configured AsyncClient
        """
        key = get_api_key("OPENROUTER_API_KEY", api_key)

        return httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
        )
