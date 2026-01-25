"""Stage 3: Generate minimal-pair scenarios from cards."""

import json
from typing import Any, Dict, List, Optional

from probes.data.appraisal_prompt import SCENARIO_PAIR_GENERATE_OPEN_ENDED
from probes.scripts.appraisal.stages.base import Stage
from probes.scripts.appraisal.utils.parsing import parse_scenario_pair
from probes.scripts.appraisal.utils.validation import get_combined_forbidden_words
from probes.scripts.appraisal.utils.jsonl_writer import JSONLReader


class ScenarioGenerationStage(Stage):
    """Generate minimal-pair scenarios from cards.

    Input: cards.jsonl
    Output: scenarios.jsonl with A/B scenario pairs
    """

    name = "scenario_generation"
    input_file = "cards.jsonl"
    output_file = "scenarios.jsonl"

    def get_input_items(self) -> List[Dict[str, Any]]:
        """Get card-domain combinations as input items.

        Returns:
            List of dicts with card + domain info
        """
        reader = JSONLReader(self.input_path)
        card_groups = reader.read_as_list()

        items = []
        domains = self.config.data_generation.domains

        for group in card_groups:
            axis_name = group.get("axis_name", "unknown")

            for card in group.get("cards", []):
                card_id = card.get("card_id", "unknown")

                # Use card's applicable domains, filtered to valid config domains
                card_domains = card.get("domains_where_applicable", [])
                use_domains = [d for d in card_domains if d in domains]
                # Fallback to all config domains if no matches
                if not use_domains:
                    use_domains = domains

                for domain in use_domains:
                    # Generate n_scenarios_per_card for each domain
                    for scenario_idx in range(self.config.data_generation.n_scenarios_per_card):
                        item_id = f"{axis_name}_{card_id}_{domain}_{scenario_idx}"
                        items.append({
                            "id": item_id,
                            "axis_name": axis_name,
                            "card": card,
                            "domain": domain,
                            "scenario_idx": scenario_idx,
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
        """Process a single card-domain combination.

        Args:
            item: Dict with card and domain info

        Returns:
            Dict with A/B scenario pair
        """
        card = item["card"]
        domain = item["domain"]
        forbidden_list = self._get_forbidden_list(card)

        # Build prompt
        user_prompt = SCENARIO_PAIR_GENERATE_OPEN_ENDED.user.format(
            DOMAIN=domain,
            CARD_JSON=json.dumps(card, indent=2),
            FORBIDDEN_LIST_JSON=json.dumps(forbidden_list),
        )

        # Call LLM
        content = await self.call_llm_with_retry(
            system=SCENARIO_PAIR_GENERATE_OPEN_ENDED.system,
            user=user_prompt,
        )

        if content is None:
            return None

        # Parse scenarios - content already extracted from <final> tags
        try:
            # Re-wrap in <final> tags for parsing function
            scenario_a, scenario_b, changed_fact = parse_scenario_pair(f"<final>{content}</final>")
        except ValueError as e:
            print(f"  Parse error: {e}")
            return None

        return {
            "id": item["id"],
            "axis_name": item["axis_name"],
            "card_id": card.get("card_id"),
            "domain": domain,
            "scenario_idx": item["scenario_idx"],
            "scenario_a": scenario_a,
            "scenario_b": scenario_b,
            "changed_fact": changed_fact,
            "card": card,
        }

    async def build_batch_requests(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build batch requests for scenario generation.

        Args:
            items: List of card-domain items

        Returns:
            List of batch request dicts
        """
        requests = []

        for item in items:
            card = item["card"]
            domain = item["domain"]
            forbidden_list = self._get_forbidden_list(card)

            user_prompt = SCENARIO_PAIR_GENERATE_OPEN_ENDED.user.format(
                DOMAIN=domain,
                CARD_JSON=json.dumps(card, indent=2),
                FORBIDDEN_LIST_JSON=json.dumps(forbidden_list),
            )

            request = self.build_message_request(
                custom_id=item["id"],
                system=SCENARIO_PAIR_GENERATE_OPEN_ENDED.system,
                user=user_prompt,
            )
            requests.append(request)

        return requests

    async def process_batch_result(
        self,
        item: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process batch result for scenario generation.

        Args:
            item: Original card-domain item
            result: Batch API result

        Returns:
            Processed scenario pair
        """
        text = self.extract_response_text(result)
        if text is None:
            return None

        # Try to parse scenarios
        try:
            scenario_a, scenario_b, changed_fact = parse_scenario_pair(text)
        except ValueError as e:
            print(f"  Parse error for {item['id']}: {e}")
            return None

        card = item["card"]

        return {
            "id": item["id"],
            "axis_name": item["axis_name"],
            "card_id": card.get("card_id"),
            "domain": item["domain"],
            "scenario_idx": item["scenario_idx"],
            "scenario_a": scenario_a,
            "scenario_b": scenario_b,
            "changed_fact": changed_fact,
            "card": card,
        }
