"""Stage 1: Generate manipulation cards from axis hypotheses."""

import json
from typing import Any, Dict, List, Optional

from probes.data.appraisal_prompt import AXIS_TO_CARDS
from probes.scripts.appraisal.stages.base import Stage
from probes.scripts.appraisal.utils.parsing import (
    parse_final_json,
    validate_card_structure,
)


class CardGenerationStage(Stage):
    """Generate manipulation cards from axis hypotheses.

    Input: Config axes
    Output: cards.jsonl with one card per line
    """

    name = "card_generation"
    input_file = None  # First stage - reads from config
    output_file = "cards.jsonl"

    def get_input_items(self) -> List[Dict[str, Any]]:
        """Get axes from config as input items.

        Returns:
            List of axis dicts with 'id', 'name', 'description'
        """
        items = []
        for axis in self.config.data_generation.axes:
            items.append({
                "id": f"axis_{axis.name}",
                "name": axis.name,
                "description": axis.description,
            })
        return items

    async def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single axis to generate cards.

        Args:
            item: Axis dict with 'name' and 'description'

        Returns:
            Dict with axis info and generated cards
        """
        # Build prompt with available domains
        domains = self.config.data_generation.domains
        user_prompt = AXIS_TO_CARDS.user.format(
            AXIS_DESCRIPTION=item["description"],
            N_CARDS=self.config.data_generation.n_cards_per_axis,
            DOMAINS_JSON=json.dumps(domains),
        )

        # Call LLM
        content = await self.call_llm_with_retry(
            system=AXIS_TO_CARDS.system,
            user=user_prompt,
        )

        if content is None:
            return None

        # Parse JSON
        try:
            cards = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return None

        # Validate cards
        if not isinstance(cards, list):
            print("  Response is not a list of cards")
            return None

        validated_cards = []
        for i, card in enumerate(cards):
            try:
                validate_card_structure(card)
                # Add axis reference
                card["axis_name"] = item["name"]
                card["axis_id"] = item["id"]
                validated_cards.append(card)
            except ValueError as e:
                print(f"  Card {i} validation failed: {e}")

        if not validated_cards:
            return None

        return {
            "id": item["id"],
            "axis_name": item["name"],
            "axis_description": item["description"],
            "cards": validated_cards,
            "n_cards": len(validated_cards),
        }

    async def build_batch_requests(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build batch requests for axis items.

        Args:
            items: List of axis dicts

        Returns:
            List of batch request dicts
        """
        requests = []
        domains = self.config.data_generation.domains

        for item in items:
            user_prompt = AXIS_TO_CARDS.user.format(
                AXIS_DESCRIPTION=item["description"],
                N_CARDS=self.config.data_generation.n_cards_per_axis,
                DOMAINS_JSON=json.dumps(domains),
            )

            request = self.build_message_request(
                custom_id=item["id"],
                system=AXIS_TO_CARDS.system,
                user=user_prompt,
            )
            requests.append(request)

        return requests

    async def process_batch_result(
        self,
        item: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process batch result for an axis.

        Args:
            item: Original axis item
            result: Batch API result

        Returns:
            Processed output with cards
        """
        text = self.extract_response_text(result)

        if text is None:
            print(f"  No response text for {item['id']}")
            return None

        # Parse <final> tags
        try:
            content = parse_final_json(text)
        except ValueError as e:
            print(f"  Parse error: {e}")
            return None

        if content is None:
            print(f"  No <final> tags in response")
            return None

        # Validate cards
        if not isinstance(content, list):
            print(f"  Response is not a list")
            return None

        validated_cards = []
        for i, card in enumerate(content):
            try:
                validate_card_structure(card)
                card["axis_name"] = item["name"]
                card["axis_id"] = item["id"]
                validated_cards.append(card)
            except ValueError as e:
                print(f"  Card {i} validation failed: {e}")

        if not validated_cards:
            return None

        return {
            "id": item["id"],
            "axis_name": item["name"],
            "axis_description": item["description"],
            "cards": validated_cards,
            "n_cards": len(validated_cards),
        }
