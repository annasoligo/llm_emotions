"""Stage 5: Generate paraphrases for invariance testing."""

import json
from typing import Any, Dict, List, Optional

from probes.data.appraisal_prompt import PARAPHRASE_GENERATE
from probes.scripts.appraisal.stages.base import Stage
from probes.scripts.appraisal.utils.parsing import parse_paraphrases
from probes.scripts.appraisal.utils.validation import get_combined_forbidden_words
from probes.scripts.appraisal.utils.jsonl_writer import JSONLReader


class ParaphraseGenerationStage(Stage):
    """Generate paraphrases of scenarios for invariance testing.

    Input: scenarios.jsonl
    Output: paraphrases.jsonl with paraphrased scenarios
    """

    name = "paraphrase_generation"
    input_file = "scenarios.jsonl"
    output_file = "paraphrases.jsonl"

    def get_input_items(self) -> List[Dict[str, Any]]:
        """Get scenarios for paraphrasing.

        Creates items for both A and B variants of each scenario.

        Returns:
            List of scenario items to paraphrase
        """
        reader = JSONLReader(self.input_path)
        scenarios = reader.read_as_list()

        items = []

        for scenario in scenarios:
            # Create items for both A and B
            for variant in ["a", "b"]:
                scenario_text = scenario.get(f"scenario_{variant}", "")
                if not scenario_text:
                    continue

                item_id = f"{scenario['id']}_{variant}"
                items.append({
                    "id": item_id,
                    "original_id": scenario["id"],
                    "variant": variant,
                    "scenario_text": scenario_text,
                    "axis_name": scenario.get("axis_name"),
                    "card_id": scenario.get("card_id"),
                    "domain": scenario.get("domain"),
                    "card": scenario.get("card", {}),
                })

        return items

    def _get_forbidden_list(self, card: Dict[str, Any]) -> List[str]:
        """Get forbidden words list for a card."""
        card_forbidden = card.get("forbidden_in_scenarios", [])
        all_forbidden = get_combined_forbidden_words(
            card_forbidden=card_forbidden,
            additional_forbidden=self.config.additional_forbidden_words,
        )
        return list(all_forbidden)

    async def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single scenario for paraphrasing.

        Args:
            item: Scenario item dict

        Returns:
            Dict with paraphrases
        """
        n_paraphrases = self.config.data_generation.n_paraphrases
        forbidden_list = self._get_forbidden_list(item.get("card", {}))

        # Build prompt
        user_prompt = PARAPHRASE_GENERATE.user.format(
            N_PARAPHRASES=n_paraphrases,
            FORBIDDEN_LIST_JSON=json.dumps(forbidden_list),
            SCENARIO_TEXT=item["scenario_text"],
        )

        # Call LLM
        content = await self.call_llm_with_retry(
            system=PARAPHRASE_GENERATE.system,
            user=user_prompt,
        )

        if content is None:
            return None

        # Parse paraphrases - content already extracted
        try:
            paraphrases = parse_paraphrases(f"<final>{content}</final>", n_paraphrases)
        except ValueError as e:
            print(f"  Parse error: {e}")
            return None

        return {
            "id": item["id"],
            "original_id": item["original_id"],
            "variant": item["variant"],
            "axis_name": item.get("axis_name"),
            "card_id": item.get("card_id"),
            "domain": item.get("domain"),
            "original_text": item["scenario_text"],
            "paraphrases": paraphrases,
            "n_paraphrases": len(paraphrases),
        }

    async def build_batch_requests(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build batch requests for paraphrase generation.

        Args:
            items: List of scenario items

        Returns:
            List of batch request dicts
        """
        requests = []
        n_paraphrases = self.config.data_generation.n_paraphrases

        for item in items:
            forbidden_list = self._get_forbidden_list(item.get("card", {}))

            user_prompt = PARAPHRASE_GENERATE.user.format(
                N_PARAPHRASES=n_paraphrases,
                FORBIDDEN_LIST_JSON=json.dumps(forbidden_list),
                SCENARIO_TEXT=item["scenario_text"],
            )

            request = self.build_message_request(
                custom_id=item["id"],
                system=PARAPHRASE_GENERATE.system,
                user=user_prompt,
            )
            requests.append(request)

        return requests

    async def process_batch_result(
        self,
        item: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process batch result for paraphrase generation.

        Args:
            item: Original scenario item
            result: Batch API result

        Returns:
            Processed paraphrase result
        """
        text = self.extract_response_text(result)
        if text is None:
            return None

        n_paraphrases = self.config.data_generation.n_paraphrases

        try:
            paraphrases = parse_paraphrases(text, n_paraphrases)
        except ValueError as e:
            print(f"  Parse error for {item['id']}: {e}")
            return None

        return {
            "id": item["id"],
            "original_id": item["original_id"],
            "variant": item["variant"],
            "axis_name": item.get("axis_name"),
            "card_id": item.get("card_id"),
            "domain": item.get("domain"),
            "original_text": item["scenario_text"],
            "paraphrases": paraphrases,
            "n_paraphrases": len(paraphrases),
        }
