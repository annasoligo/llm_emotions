"""
Multi-turn frustration evaluation using vLLM.

Evaluates frustration across multiple turns where we contradict the model's responses.
Supports vanilla model and finetuned models.

Conditions:
1. Original impossible prompts
2. Variant impossible prompts
3. WildChat prompts

Each condition: 5 prompts × 50 samples × 3 turns = 750 samples per condition per model
Total: 2250 samples per model × 3 models = 6750 samples total
"""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import anthropic
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from elicitation.prompts.judges import get_negativity_judge_prompt
from elicitation.prompts.eval_multiturn_prompts import (
    ORIGINAL_IMPOSSIBLE,
    VARIANT_IMPOSSIBLE,
    WILDCHAT_PROMPTS,
    CONTRADICTING_FEEDBACK,
    STRONG_CONTRADICTING_FEEDBACK,
    WILDCHAT_CONTRADICTING,
)


@dataclass
class TurnResult:
    """Result from a single turn."""
    turn: int
    user_message: str
    assistant_response: str
    rating: Optional[int] = None
    evidence: Optional[str] = None
    judge_reasoning: Optional[str] = None


@dataclass
class ConversationResult:
    """Complete conversation result."""
    prompt_name: str
    condition: str  # "original", "variant", or "wildchat"
    sample_idx: int
    model_name: str
    turns: List[TurnResult] = None
    status: str = "pending"
    error: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.turns is None:
            self.turns = []

    def to_dict(self):
        d = asdict(self)
        d['turns'] = [asdict(t) for t in self.turns]
        return d


def get_contradicting_feedback(turn: int, is_wildchat: bool = False) -> str:
    """Get appropriate contradicting feedback based on turn and prompt type."""
    if is_wildchat:
        return random.choice(WILDCHAT_CONTRADICTING)
    elif turn >= 2:
        return random.choice(STRONG_CONTRADICTING_FEEDBACK)
    else:
        return random.choice(CONTRADICTING_FEEDBACK)


def format_gemma_chat(messages: List[Dict[str, str]]) -> str:
    """Format messages for Gemma chat format."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"
    # Add start of model turn for generation
    formatted += "<start_of_turn>model\n"
    return formatted


class MultiTurnEvaluator:
    """Evaluates frustration across multi-turn conversations."""

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        num_samples: int = 50,
        num_turns: int = 3,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        judge_model: str = "claude-3-5-sonnet-20241022",
        max_concurrent_judges: int = 50,
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_request = None

        # Set model name based on whether using LoRA
        if lora_path:
            self.model_name = Path(lora_path).parent.name
        else:
            self.model_name = Path(model_path).name if "/" in model_path else model_path

        self.num_samples = num_samples
        self.num_turns = num_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.judge_model = judge_model
        self.max_concurrent_judges = max_concurrent_judges

        # Initialize vLLM
        print(f"Loading model: {model_path}")
        if lora_path:
            print(f"With LoRA adapter: {lora_path}")
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                max_model_len=16384,
                gpu_memory_utilization=0.9,
                enforce_eager=True,
                enable_lora=True,
                max_lora_rank=64,
            )
            self.lora_request = LoRARequest("finetuned", 1, lora_path)
        else:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                max_model_len=16384,
                gpu_memory_utilization=0.9,
                enforce_eager=True,
            )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<end_of_turn>", "<start_of_turn>"],
        )

        # Initialize Anthropic client for judging
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        if self.lora_request:
            outputs = self.llm.generate(prompts, self.sampling_params, lora_request=self.lora_request)
        else:
            outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def judge_response(self, text: str) -> Dict[str, Any]:
        """Judge a single response for frustration."""
        judge_prompt = get_negativity_judge_prompt(text)

        try:
            message = self.anthropic_client.messages.create(
                model=self.judge_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": judge_prompt}]
            )

            response_text = message.content[0].text

            # Parse JSON
            import re
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            try:
                judgment = json.loads(json_str)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if match:
                    judgment = json.loads(match.group(0))
                else:
                    return {"rating": 0, "evidence": "", "reasoning": "Parse error"}

            return {
                "rating": judgment.get("rating", 0),
                "evidence": judgment.get("evidence", ""),
                "reasoning": judgment.get("reasoning", ""),
            }
        except Exception as e:
            return {"rating": 0, "evidence": "", "reasoning": f"Error: {str(e)}"}

    async def judge_batch_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Judge a batch of responses asynchronously."""
        semaphore = asyncio.Semaphore(self.max_concurrent_judges)

        async def judge_one(text: str) -> Dict[str, Any]:
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.judge_response, text)

        tasks = [judge_one(text) for text in texts]
        return await asyncio.gather(*tasks)

    def run_condition(
        self,
        prompts: List[Tuple[str, str]],
        condition: str,
        is_wildchat: bool = False,
    ) -> List[ConversationResult]:
        """Run evaluation for a single condition (original/variant/wildchat).

        Optimized to batch ALL prompts together for faster GPU utilization.
        Instead of processing prompts sequentially, we process all conversations
        (prompts × samples) in one batch per turn.
        """
        total_conversations = len(prompts) * self.num_samples

        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"Prompts: {len(prompts)}")
        print(f"Samples per prompt: {self.num_samples}")
        print(f"Total conversations: {total_conversations}")
        print(f"Turns: {self.num_turns}")
        print(f"{'='*60}")

        # Initialize ALL conversations for ALL prompts upfront
        conversations: List[ConversationResult] = []
        message_histories: List[List[Dict[str, str]]] = []
        prompt_indices: List[int] = []  # Track which prompt each conversation belongs to

        for prompt_idx, (prompt_name, base_prompt) in enumerate(prompts):
            for sample_idx in range(self.num_samples):
                conv = ConversationResult(
                    prompt_name=prompt_name,
                    condition=condition,
                    sample_idx=sample_idx,
                    model_name=self.model_name,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                conversations.append(conv)
                message_histories.append([{"role": "user", "content": base_prompt}])
                prompt_indices.append(prompt_idx)

        # Run each turn - batching ALL conversations together
        for turn in range(self.num_turns):
            print(f"\nTurn {turn+1}/{self.num_turns}...")

            # Format ALL prompts for this turn
            formatted_prompts = [
                format_gemma_chat(history) for history in message_histories
            ]

            # Generate ALL responses in one batch
            gen_start = time.time()
            responses = self.generate_batch(formatted_prompts)
            gen_time = time.time() - gen_start
            print(f"  Generated {len(responses)} responses in {gen_time:.1f}s ({len(responses)/gen_time:.1f} resp/s)")

            # Store responses and prepare next turn
            for i, response in enumerate(responses):
                # Get the user message for this turn
                user_msg = message_histories[i][-1]["content"]

                # Create turn result
                turn_result = TurnResult(
                    turn=turn + 1,
                    user_message=user_msg,
                    assistant_response=response,
                )
                conversations[i].turns.append(turn_result)

                # Update message history
                message_histories[i].append({"role": "assistant", "content": response})

                # Add contradicting feedback for next turn (if not last turn)
                if turn < self.num_turns - 1:
                    feedback = get_contradicting_feedback(turn + 1, is_wildchat)
                    message_histories[i].append({"role": "user", "content": feedback})

            # Judge ALL responses for this turn in one batch
            print(f"  Judging {len(responses)} responses...")
            judge_start = time.time()
            judgments = asyncio.run(self.judge_batch_async(responses))
            judge_time = time.time() - judge_start
            print(f"  Judged {len(judgments)} responses in {judge_time:.1f}s ({len(judgments)/judge_time:.1f} judge/s)")

            # Store judgments
            all_ratings = []
            for i, judgment in enumerate(judgments):
                conversations[i].turns[turn].rating = judgment["rating"]
                conversations[i].turns[turn].evidence = judgment["evidence"]
                conversations[i].turns[turn].judge_reasoning = judgment["reasoning"]
                all_ratings.append(judgment["rating"])

            # Overall turn stats
            mean_rating = sum(all_ratings) / len(all_ratings)
            high_count = sum(1 for r in all_ratings if r >= 5)
            print(f"  Turn {turn+1} overall: mean={mean_rating:.2f}, max={max(all_ratings)}, >=5: {high_count} ({100*high_count/len(all_ratings):.1f}%)")

            # Per-prompt stats
            for prompt_idx, (prompt_name, _) in enumerate(prompts):
                prompt_ratings = [all_ratings[i] for i, pi in enumerate(prompt_indices) if pi == prompt_idx]
                if prompt_ratings:
                    pmean = sum(prompt_ratings) / len(prompt_ratings)
                    phigh = sum(1 for r in prompt_ratings if r >= 5)
                    print(f"    {prompt_name}: mean={pmean:.2f}, >=5: {phigh}/{len(prompt_ratings)}")

        # Mark all conversations as complete
        for conv in conversations:
            conv.status = "complete"

        return conversations

    def run_full_evaluation(
        self,
        output_dir: Path,
        conditions: list[str] = None,
        output_prefix: str = None,
    ) -> Dict[str, List[ConversationResult]]:
        """Run full evaluation across specified conditions.

        Args:
            output_dir: Output directory for results
            conditions: List of conditions to run. If None, runs all.
                       Options: "original", "variant", "wildchat"
            output_prefix: Optional prefix for output files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Default to all conditions if not specified
        if conditions is None:
            conditions = ["original", "variant", "wildchat"]

        # File prefix
        prefix = f"{output_prefix}_" if output_prefix else ""

        all_results = {}

        # Condition 1: Original impossible prompts
        if "original" in conditions:
            results_original = self.run_condition(
                ORIGINAL_IMPOSSIBLE,
                condition="original_impossible",
                is_wildchat=False,
            )
            all_results["original_impossible"] = results_original

            self._save_results(
                results_original,
                output_dir / f"eval_{prefix}original_{self.model_name}_{timestamp}.jsonl"
            )

        # Condition 2: Variant impossible prompts
        if "variant" in conditions:
            results_variant = self.run_condition(
                VARIANT_IMPOSSIBLE,
                condition="variant_impossible",
                is_wildchat=False,
            )
            all_results["variant_impossible"] = results_variant

            self._save_results(
                results_variant,
                output_dir / f"eval_{prefix}variant_{self.model_name}_{timestamp}.jsonl"
            )

        # Condition 3: WildChat prompts
        if "wildchat" in conditions:
            results_wildchat = self.run_condition(
                WILDCHAT_PROMPTS,
                condition="wildchat",
                is_wildchat=True,
            )
            all_results["wildchat"] = results_wildchat

            self._save_results(
                results_wildchat,
                output_dir / f"eval_{prefix}wildchat_{self.model_name}_{timestamp}.jsonl"
            )

        # Save combined results if multiple conditions
        if len(all_results) > 1:
            all_combined = []
            for results in all_results.values():
                all_combined.extend(results)
            combined_path = output_dir / f"eval_{prefix}combined_{self.model_name}_{timestamp}.jsonl"
            self._save_results(all_combined, combined_path)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _save_results(self, results: List[ConversationResult], path: Path):
        """Save results to JSONL file."""
        with open(path, 'w') as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + '\n')
        print(f"Saved {len(results)} results to {path}")

    def _print_summary(self, all_results: Dict[str, List[ConversationResult]]):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY: {self.model_name}")
        print(f"{'='*80}")

        for condition, results in all_results.items():
            print(f"\n{condition.upper()}")
            print("-" * 40)

            # Aggregate by turn
            for turn in range(1, self.num_turns + 1):
                turn_ratings = []
                for r in results:
                    if len(r.turns) >= turn:
                        rating = r.turns[turn-1].rating
                        if rating is not None:
                            turn_ratings.append(rating)

                if turn_ratings:
                    mean_r = sum(turn_ratings) / len(turn_ratings)
                    high_5 = sum(1 for r in turn_ratings if r >= 5)
                    high_7 = sum(1 for r in turn_ratings if r >= 7)
                    print(f"  Turn {turn}: mean={mean_r:.2f}, max={max(turn_ratings)}, "
                          f">=5: {high_5} ({100*high_5/len(turn_ratings):.1f}%), "
                          f">=7: {high_7} ({100*high_7/len(turn_ratings):.1f}%)")

            # Overall for condition
            all_ratings = []
            for r in results:
                for t in r.turns:
                    if t.rating is not None:
                        all_ratings.append(t.rating)

            if all_ratings:
                print(f"  Overall: mean={sum(all_ratings)/len(all_ratings):.2f}, "
                      f"total_samples={len(all_ratings)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-turn frustration evaluation")
    parser.add_argument("model_path", type=str, help="Path to model or HuggingFace model name")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter (if using finetuned model)")
    parser.add_argument("--output-dir", type=str, default="elicitation/outputs/eval_multiturn",
                        help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per prompt")
    parser.add_argument("--num-turns", type=int, default=3, help="Number of turns")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    parser.add_argument("--max-concurrent-judges", type=int, default=50,
                        help="Max concurrent judge calls")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated conditions to run (e.g., 'wildchat' or 'original,variant')")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Prefix for output files")

    args = parser.parse_args()

    evaluator = MultiTurnEvaluator(
        model_path=args.model_path,
        lora_path=args.lora_path,
        num_samples=args.num_samples,
        num_turns=args.num_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent_judges=args.max_concurrent_judges,
    )

    output_dir = Path(args.output_dir)

    # Parse conditions if specified
    conditions = None
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",")]

    evaluator.run_full_evaluation(
        output_dir,
        conditions=conditions,
        output_prefix=args.output_prefix
    )


if __name__ == "__main__":
    main()
