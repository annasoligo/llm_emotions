"""Abstract base class for pipeline stages."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
import json

from probes.data.api_clients import AnthropicBatchClient, get_api_key
from probes.scripts.appraisal.config import AppraisalConfig
from probes.scripts.appraisal.utils.progress import StateManager
from probes.scripts.appraisal.utils.jsonl_writer import JSONLWriter
from probes.scripts.appraisal.utils.parsing import parse_final_tags, parse_final_json


class Stage(ABC):
    """Abstract base class for pipeline stages.

    Each stage:
    1. Reads input from previous stage output (or config)
    2. Processes items (typically via LLM calls)
    3. Writes output to JSONL file
    4. Tracks progress for resume support

    Subclasses must implement:
    - name: Stage identifier
    - input_file: Name of input JSONL file (or None for first stage)
    - output_file: Name of output JSONL file
    - process_item(): Process a single item
    - get_input_items(): Get list of items to process
    """

    # Class attributes to override
    name: str = "base"
    input_file: Optional[str] = None
    output_file: str = "output.jsonl"

    def __init__(
        self,
        config: AppraisalConfig,
        state_manager: StateManager,
    ):
        """Initialize stage.

        Args:
            config: Pipeline configuration
            state_manager: State manager for progress tracking
        """
        self.config = config
        self.state = state_manager

        # Setup paths
        self.output_path = config.get_output_path(self.output_file)

        if self.input_file:
            self.input_path = config.get_output_path(self.input_file)
        else:
            self.input_path = None

        # Initialize writer
        self.writer = JSONLWriter(self.output_path)

        # Batch client (lazily initialized)
        self._batch_client: Optional[AnthropicBatchClient] = None

    @property
    def batch_client(self) -> AnthropicBatchClient:
        """Get or create batch client."""
        if self._batch_client is None:
            api_key = get_api_key("ANTHROPIC_API_KEY")
            self._batch_client = AnthropicBatchClient(api_key)
        return self._batch_client

    @abstractmethod
    def get_input_items(self) -> List[Dict[str, Any]]:
        """Get list of items to process.

        Returns:
            List of input item dicts
        """
        pass

    @abstractmethod
    async def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item.

        Args:
            item: Input item dict

        Returns:
            Processed output dict, or None if processing failed
        """
        pass

    def get_pending_items(self) -> List[Dict[str, Any]]:
        """Get items that haven't been processed yet.

        Uses state manager to filter out completed items.

        Returns:
            List of pending items
        """
        all_items = self.get_input_items()
        completed_ids = self.state.get_completed_ids(self.name)

        pending = []
        for item in all_items:
            item_id = item.get("id", str(hash(json.dumps(item, sort_keys=True))))
            if item_id not in completed_ids:
                pending.append(item)

        return pending

    async def run(self, concurrency: int = 1) -> bool:
        """Run the stage.

        Processes all pending items with progress tracking.

        Args:
            concurrency: Number of concurrent requests (default 1 for sequential)

        Returns:
            True if stage completed successfully
        """
        print(f"\n{'='*60}")
        print(f"STAGE: {self.name}")
        print(f"{'='*60}")

        # Get pending items
        pending = self.get_pending_items()
        total = len(self.get_input_items())
        completed = total - len(pending)

        print(f"Items: {completed}/{total} completed, {len(pending)} pending")
        if concurrency > 1:
            print(f"Concurrency: {concurrency}")

        if not pending:
            print("No items to process - stage already complete")
            return True

        # Process items
        success_count = 0
        error_count = 0

        if concurrency == 1:
            # Sequential processing
            for i, item in enumerate(pending):
                item_id = item.get("id", str(i))

                try:
                    print(f"\n[{i+1}/{len(pending)}] Processing {item_id}...")

                    result = await self.process_item(item)

                    if result is not None:
                        self.writer.write(result)
                        self.state.mark_completed(self.name, item_id)
                        success_count += 1
                        print(f"  Done")
                    else:
                        error_count += 1
                        print(f"  Failed (no result)")

                except Exception as e:
                    error_count += 1
                    print(f"  Error ({type(e).__name__}): {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Concurrent processing with semaphore
            semaphore = asyncio.Semaphore(concurrency)
            results_lock = asyncio.Lock()
            progress = {"completed": 0, "success": 0, "error": 0}

            async def process_with_semaphore(item, idx):
                item_id = item.get("id", str(idx))
                async with semaphore:
                    try:
                        result = await self.process_item(item)

                        async with results_lock:
                            progress["completed"] += 1
                            if result is not None:
                                self.writer.write(result)
                                self.state.mark_completed(self.name, item_id)
                                progress["success"] += 1
                                print(f"  [{progress['completed']}/{len(pending)}] {item_id} - Done")
                            else:
                                progress["error"] += 1
                                print(f"  [{progress['completed']}/{len(pending)}] {item_id} - Failed")

                    except Exception as e:
                        async with results_lock:
                            progress["completed"] += 1
                            progress["error"] += 1
                            print(f"  [{progress['completed']}/{len(pending)}] {item_id} - Error: {e}")

            # Create all tasks
            tasks = [process_with_semaphore(item, i) for i, item in enumerate(pending)]
            await asyncio.gather(*tasks)

            success_count = progress["success"]
            error_count = progress["error"]

        # Summary
        print(f"\n{'-'*40}")
        print(f"Stage {self.name} complete:")
        print(f"  Success: {success_count}")
        print(f"  Errors: {error_count}")

        return error_count == 0

    async def run_batch(self, batch_size: int = 50) -> bool:
        """Run stage using batch API for efficiency.

        Groups items into batches and processes via Anthropic Batch API.

        Args:
            batch_size: Number of items per batch

        Returns:
            True if stage completed successfully
        """
        print(f"\n{'='*60}")
        print(f"STAGE: {self.name} (Batch Mode)")
        print(f"{'='*60}")

        # Get pending items
        pending = self.get_pending_items()
        total = len(self.get_input_items())
        completed = total - len(pending)

        print(f"Items: {completed}/{total} completed, {len(pending)} pending")

        if not pending:
            print("No items to process - stage already complete")
            return True

        # Process in batches
        success_count = 0
        error_count = 0

        for batch_start in range(0, len(pending), batch_size):
            batch_end = min(batch_start + batch_size, len(pending))
            batch = pending[batch_start:batch_end]

            print(f"\nProcessing batch {batch_start//batch_size + 1} "
                  f"({len(batch)} items)...")

            try:
                # Build batch requests
                requests = await self.build_batch_requests(batch)

                if not requests:
                    print("  No valid requests in batch")
                    continue

                # Submit and wait
                results = await self.batch_client.submit_and_wait(requests)

                # Process results
                for item in batch:
                    item_id = item.get("id", str(hash(json.dumps(item, sort_keys=True))))

                    if item_id in results:
                        result_data = results[item_id]

                        try:
                            processed = await self.process_batch_result(item, result_data)

                            if processed is not None:
                                self.writer.write(processed)
                                self.state.mark_completed(self.name, item_id)
                                success_count += 1
                            else:
                                error_count += 1

                        except Exception as e:
                            print(f"  Error processing {item_id}: {e}")
                            error_count += 1
                    else:
                        print(f"  No result for {item_id}")
                        error_count += 1

            except Exception as e:
                print(f"  Batch error: {e}")
                error_count += len(batch)

        # Summary
        print(f"\n{'-'*40}")
        print(f"Stage {self.name} complete:")
        print(f"  Success: {success_count}")
        print(f"  Errors: {error_count}")

        return error_count == 0

    async def build_batch_requests(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build batch API requests for items.

        Override in subclasses to customize request building.

        Args:
            items: Items to build requests for

        Returns:
            List of batch request dicts
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclass must implement build_batch_requests")

    async def process_batch_result(
        self,
        item: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single result from batch API.

        Override in subclasses to customize result processing.

        Args:
            item: Original input item
            result: Result from batch API

        Returns:
            Processed output dict, or None if failed
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclass must implement process_batch_result")

    def extract_response_text(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract text content from batch API result.

        Args:
            result: Result dict from batch API

        Returns:
            Response text, or None if extraction failed
        """
        try:
            # Navigate batch result structure
            if result.get("result", {}).get("type") == "succeeded":
                message = result["result"]["message"]
                content = message.get("content", [])

                # Find text block
                for block in content:
                    if block.get("type") == "text":
                        return block.get("text", "")

            return None

        except (KeyError, TypeError):
            return None

    def build_message_request(
        self,
        custom_id: str,
        system: str,
        user: str,
    ) -> Dict[str, Any]:
        """Build a batch request for a single message.

        Args:
            custom_id: Unique ID for this request
            system: System prompt
            user: User message

        Returns:
            Batch request dict
        """
        return {
            "custom_id": custom_id,
            "params": {
                "model": self.config.claude_model,
                "max_tokens": self.config.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        }

    async def call_llm_with_retry(
        self,
        system: str,
        user: str,
        max_retries: int = 3,
        max_rate_limit_retries: int = 5,
    ) -> Optional[str]:
        """Call LLM with retry logic for <final> tag parsing and rate limits.

        If response doesn't have <final> tags, sends repair prompt.
        Handles 429 rate limit errors with exponential backoff.

        Args:
            system: System prompt
            user: User message
            max_retries: Maximum retry attempts for parsing failures
            max_rate_limit_retries: Maximum retries for rate limit errors

        Returns:
            Content from <final> tags, or None if all retries failed
        """
        import anthropic

        client = anthropic.AsyncAnthropic()

        original_user = user
        rate_limit_retries = 0

        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=self.config.claude_model,
                    max_tokens=self.config.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )

                text = response.content[0].text

                # Try to parse <final> tags
                content = parse_final_tags(text)

                if content is not None:
                    return content

                # No <final> tags - send repair prompt
                if attempt < max_retries - 1:
                    print(f"  No <final> tags, retrying ({attempt + 1}/{max_retries})...")
                    user = (
                        "Your previous response did not include <final> tags. "
                        "Please provide your answer wrapped in <final>...</final> tags.\n\n"
                        f"Original request:\n{original_user}"
                    )

            except anthropic.RateLimitError as e:
                rate_limit_retries += 1
                if rate_limit_retries <= max_rate_limit_retries:
                    # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                    wait_time = 30 * (2 ** (rate_limit_retries - 1))
                    print(f"  Rate limit hit, waiting {wait_time}s ({rate_limit_retries}/{max_rate_limit_retries})...")
                    await asyncio.sleep(wait_time)
                    # Don't count rate limit retries against max_retries
                    attempt -= 1
                else:
                    print(f"  Rate limit retries exhausted")
                    raise

            except anthropic.APIStatusError as e:
                if e.status_code == 429:
                    # Handle 429 same as RateLimitError
                    rate_limit_retries += 1
                    if rate_limit_retries <= max_rate_limit_retries:
                        wait_time = 30 * (2 ** (rate_limit_retries - 1))
                        print(f"  Rate limit (429), waiting {wait_time}s ({rate_limit_retries}/{max_rate_limit_retries})...")
                        await asyncio.sleep(wait_time)
                        attempt -= 1
                    else:
                        raise
                elif e.status_code >= 500:
                    # Server errors - retry with shorter backoff
                    if attempt < max_retries - 1:
                        wait_time = 5 * (2 ** attempt)
                        print(f"  Server error ({e.status_code}), waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  LLM error: {e}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        return None
