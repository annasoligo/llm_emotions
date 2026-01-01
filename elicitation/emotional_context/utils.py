"""
Shared utility functions for emotional context evaluation system.
"""
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import anthropic
import requests
from tqdm.asyncio import tqdm as atqdm

# Model constants
opus_4_5 = "claude-opus-4-5-20251101"
sonnet_4_5 = "claude-sonnet-4-5-20250929"
gemma_3_27b = "google/gemma-3-27b-it"  # OpenRouter model ID

# API Keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], filepath: Path) -> None:
    """Save list of dicts to JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIClient:
    """Wrapper for both Anthropic and OpenRouter APIs with retry logic."""

    def __init__(self, max_retries: int = 3, initial_delay: float = 2.0):
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        self.max_retries = max_retries
        self.initial_delay = initial_delay

    def _is_openrouter_model(self, model: str) -> bool:
        """Check if model should use OpenRouter API."""
        return "google/" in model or "gemma" in model.lower()

    async def _generate_openrouter_async(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate using OpenRouter API."""
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, headers=headers, json=payload, timeout=120)
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _generate_anthropic_async(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate using Anthropic API."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.anthropic_client.messages.create(**kwargs)
        )

        return response.content[0].text

    async def generate_async(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate with retry logic and exponential backoff."""

        for attempt in range(self.max_retries):
            try:
                if self._is_openrouter_model(model):
                    return await self._generate_openrouter_async(
                        prompt, model, temperature, max_tokens, system_prompt
                    )
                else:
                    return await self._generate_anthropic_async(
                        prompt, model, temperature, max_tokens, system_prompt
                    )

            except (anthropic.RateLimitError, requests.exceptions.HTTPError) as e:
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (2 ** attempt)
                    logger.warning(f"Rate limit/HTTP error, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} attempts")
                    raise

            except (anthropic.APIError, requests.exceptions.RequestException) as e:
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (2 ** attempt)
                    logger.warning(f"API error: {e}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"API error after {self.max_retries} attempts: {e}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def batch_generate_async(
        self,
        prompts: List[str],
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        max_concurrent: int = 50,
        desc: str = "Generating"
    ) -> List[str]:
        """Generate multiple responses with concurrency control."""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate_async(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt
                )

        tasks = [generate_with_semaphore(p) for p in prompts]
        return await atqdm.gather(*tasks, desc=desc)


def generate_uuid() -> str:
    """Generate a UUID for record identification."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def load_stage_data(filepath: Path) -> List[Dict]:
    """Load JSONL data from a stage."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []
    return load_jsonl(filepath)


def save_stage_data(filepath: Path, data: List[Dict]) -> None:
    """Save JSONL data to a stage."""
    save_jsonl(data, filepath)
    logger.info(f"Saved {len(data)} records to {filepath}")


def checkpoint_save(filepath: Path, data: List[Dict], interval: int = 100) -> None:
    """Save checkpoint periodically during processing."""
    if len(data) % interval == 0 and len(data) > 0:
        save_stage_data(filepath, data)
        logger.info(f"Checkpoint: Saved {len(data)} records")
