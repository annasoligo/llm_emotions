#!/usr/bin/env python3
"""
Generate neutral paraphrases of emotional conversations using Batch API.

This script:
1. Loads emotional conversations from JSONL
2. Creates batch requests to neutralize each message
3. Submits to Anthropic Batch API
4. Polls for completion and saves neutral conversations

Usage:
    python -m probes.scripts.generate_neutral_conversations_batch \
        --input data/conversations2.jsonl \
        --output data/conversations2_neutral.jsonl \
        --batch_output batches/neutral_conversations_batch.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import os

import anthropic
from tqdm import tqdm


def load_conversations(input_path: Path) -> List[Dict]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(input_path, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations


def create_batch_requests(conversations: List[Dict], topic: str = None) -> List[Dict]:
    """Create batch API requests for neutralizing all messages.

    Each message in each conversation gets its own request.

    Args:
        conversations: List of conversation dicts
        topic: Optional topic override

    Returns:
        List of batch request dicts
    """
    batch_requests = []

    prompt_template = """You are generating a neutral, emotionally flat version of a conversation message.

Original message:
{original_message}

Context: This message is part of a conversation about "{topic}". The original expressed {emotion}.

Generate a NEUTRAL paraphrase that:
1. Preserves ALL semantic content and information
2. Removes emotional language (fear, anger, joy, surprise, disgust, sadness markers)
3. Uses calm, balanced, matter-of-fact tone
4. Maintains similar length and structure
5. Keeps the conversational flow natural

Guidelines:
- Replace emotional adjectives (terrified→concerned, furious→disagree, thrilled→pleased)
- Remove exclamations, strong punctuation, ALL CAPS
- Use neutral hedging (might, could, appears to) instead of certainty
- Keep factual content identical
- Don't add robotic formality - stay conversational but neutral

Output ONLY the neutral paraphrase, nothing else."""

    for conv_idx, conversation in enumerate(tqdm(conversations, desc="Creating batch requests")):
        conv_topic = topic if topic else conversation.get("topic", "unknown")

        for msg_idx, msg in enumerate(conversation["messages"]):
            # Determine emotion for this message
            if msg["role"] == "user":
                emotion = conversation["user_emotion"]
            elif msg["role"] == "assistant":
                emotion = conversation["asst_emotion"]
            else:
                continue

            # Create prompt
            prompt = prompt_template.format(
                original_message=msg["content"],
                topic=conv_topic,
                emotion=emotion,
            )

            # Create batch request
            custom_id = f"conv_{conv_idx}_msg_{msg_idx}_{msg['role']}"

            batch_request = {
                "custom_id": custom_id,
                "params": {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 500,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            }

            batch_requests.append(batch_request)

    return batch_requests


def save_batch_requests(batch_requests: List[Dict], output_path: Path) -> None:
    """Save batch requests to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')


def submit_batch(batch_file_path: Path, api_key: str = None) -> str:
    """Submit batch requests to Anthropic API.

    Returns:
        batch_id: ID of the submitted batch
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Read batch requests
    print(f"Loading batch file: {batch_file_path}")
    batch_requests = []
    with open(batch_file_path, 'r') as f:
        for line in f:
            batch_requests.append(json.loads(line))

    print(f"Submitting {len(batch_requests)} requests to Batch API...")
    batch = client.messages.batches.create(
        requests=batch_requests
    )

    print(f"Batch submitted: {batch.id}")
    print(f"Status: {batch.processing_status}")

    return batch.id


def wait_for_batch(batch_id: str, api_key: str = None, poll_interval: int = 60) -> Dict:
    """Poll batch status until completion.

    Args:
        batch_id: Batch ID to poll
        api_key: Anthropic API key
        poll_interval: Seconds between polls

    Returns:
        Final batch object
    """
    client = anthropic.Anthropic(api_key=api_key)

    print(f"\nPolling batch {batch_id}...")

    while True:
        batch = client.messages.batches.retrieve(batch_id)

        status = batch.processing_status

        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
            succeeded = counts.succeeded
            print(f"Status: {status} | Progress: {succeeded}/{total} succeeded")
        else:
            print(f"Status: {status}")

        if status == "ended":
            print("\nBatch completed!")
            print(f"  Succeeded: {batch.request_counts.succeeded}")
            print(f"  Errored: {batch.request_counts.errored}")
            print(f"  Canceled: {batch.request_counts.canceled}")
            print(f"  Expired: {batch.request_counts.expired}")
            return batch
        elif status in ["canceling", "canceled"]:
            raise ValueError(f"Batch was canceled: {batch_id}")

        time.sleep(poll_interval)


def download_batch_results(batch_id: str, output_path: Path, api_key: str = None) -> None:
    """Download batch results to file.

    Args:
        batch_id: Batch ID
        output_path: Where to save results
        api_key: Anthropic API key
    """
    client = anthropic.Anthropic(api_key=api_key)

    print(f"\nDownloading results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get results iterator
    results = client.messages.batches.results(batch_id)

    # Write to file (convert to dict and use compact JSON)
    with open(output_path, 'w') as f:
        for result in results:
            # Convert to dict and write as compact JSON
            result_dict = json.loads(result.to_json())
            f.write(json.dumps(result_dict) + '\n')

    print(f"Results saved to {output_path}")


def reconstruct_neutral_conversations(
    original_conversations: List[Dict],
    results_path: Path,
) -> List[Dict]:
    """Reconstruct neutral conversations from batch results.

    Args:
        original_conversations: Original emotional conversations
        results_path: Path to batch results JSONL

    Returns:
        List of neutral conversation dicts
    """
    # Load results
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']

            if result['result']['type'] == 'succeeded':
                message = result['result']['message']
                neutral_text = message['content'][0]['text'].strip()
                results[custom_id] = neutral_text
            else:
                # Handle error
                error = result['result']['error']
                print(f"Warning: Request {custom_id} failed: {error}")
                results[custom_id] = None

    # Reconstruct conversations
    neutral_conversations = []

    for conv_idx, conversation in enumerate(tqdm(original_conversations, desc="Reconstructing conversations")):
        neutral_messages = []

        for msg_idx, msg in enumerate(conversation["messages"]):
            custom_id = f"conv_{conv_idx}_msg_{msg_idx}_{msg['role']}"

            if custom_id in results and results[custom_id] is not None:
                neutral_content = results[custom_id]
            else:
                # Fallback: use original message if neutralization failed
                print(f"Warning: Using original message for {custom_id}")
                neutral_content = msg["content"]

            neutral_messages.append({
                "role": msg["role"],
                "content": neutral_content,
            })

        # Create neutral conversation
        neutral_conv = {
            **conversation,  # Preserve metadata
            "user_emotion": "neutral",
            "asst_emotion": "neutral",
            "messages": neutral_messages,
        }

        neutral_conversations.append(neutral_conv)

    return neutral_conversations


def save_conversations(conversations: List[Dict], output_path: Path) -> None:
    """Save conversations to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate neutral conversation paraphrases via Batch API')
    parser.add_argument('--input', type=str, required=True, help='Input conversations JSONL')
    parser.add_argument('--output', type=str, required=True, help='Output neutral conversations JSONL')
    parser.add_argument('--batch_requests', type=str, default='batches/neutral_requests.jsonl',
                        help='Where to save batch requests')
    parser.add_argument('--batch_results', type=str, default='batches/neutral_results.jsonl',
                        help='Where to save batch results')
    parser.add_argument('--batch_id', type=str, help='Resume existing batch (skip submission)')
    parser.add_argument('--poll_interval', type=int, default=60, help='Seconds between status polls')
    parser.add_argument('--skip_submit', action='store_true', help='Only create requests, don\'t submit')

    args = parser.parse_args()

    print("="*80)
    print("GENERATE NEUTRAL CONVERSATIONS VIA BATCH API")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()

    # Load conversations
    print("Loading conversations...")
    conversations = load_conversations(Path(args.input))
    print(f"Loaded {len(conversations)} conversations")

    total_messages = sum(len(c["messages"]) for c in conversations)
    print(f"Total messages to neutralize: {total_messages}")
    print()

    # Create batch requests
    print("Creating batch requests...")
    batch_requests = create_batch_requests(conversations)
    print(f"Created {len(batch_requests)} batch requests")
    print()

    # Save batch requests
    batch_requests_path = Path(args.batch_requests)
    print(f"Saving batch requests to {batch_requests_path}...")
    save_batch_requests(batch_requests, batch_requests_path)
    print("Saved!")
    print()

    if args.skip_submit:
        print("Skipping submission (--skip_submit flag)")
        return

    # Submit or resume batch
    if args.batch_id:
        print(f"Resuming existing batch: {args.batch_id}")
        batch_id = args.batch_id
    else:
        print("Submitting batch to Anthropic API...")
        batch_id = submit_batch(batch_requests_path)
        print()

    # Wait for completion
    batch = wait_for_batch(batch_id, poll_interval=args.poll_interval)
    print()

    # Download results
    batch_results_path = Path(args.batch_results)
    download_batch_results(batch_id, batch_results_path)
    print()

    # Reconstruct neutral conversations
    print("Reconstructing neutral conversations...")
    neutral_conversations = reconstruct_neutral_conversations(
        conversations, batch_results_path
    )
    print(f"Reconstructed {len(neutral_conversations)} neutral conversations")
    print()

    # Save output
    output_path = Path(args.output)
    print(f"Saving to {output_path}...")
    save_conversations(neutral_conversations, output_path)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
