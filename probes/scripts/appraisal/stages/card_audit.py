"""Stage 2: Audit and rewrite manipulation cards."""

import json
from typing import Any, Dict, List, Optional

from probes.data.appraisal_prompt import (
    CARDS_AUDIT_AND_REWRITE,
    BANNED_TONE_WORDS_DEFAULT,
    BANNED_AXIS_VOCAB_DEFAULT,
    BANNED_URGENCY_STAKES_WORDS_DEFAULT,
)
from probes.scripts.appraisal.stages.base import Stage
from probes.scripts.appraisal.utils.parsing import parse_final_json
from probes.scripts.appraisal.utils.jsonl_writer import JSONLReader


class CardAuditStage(Stage):
    """Audit manipulation cards and rewrite failures.

    Input: cards.jsonl
    Output: cards_audited.jsonl with audit decisions
    """

    name = "card_audit"
    input_file = "cards.jsonl"
    output_file = "cards_audited.jsonl"

    def _get_banned_words_json(self) -> str:
        """Get combined banned words as JSON string."""
        all_banned = (
            BANNED_TONE_WORDS_DEFAULT +
            BANNED_AXIS_VOCAB_DEFAULT +
            BANNED_URGENCY_STAKES_WORDS_DEFAULT +
            self.config.additional_forbidden_words
        )
        return json.dumps(all_banned)

    def get_input_items(self) -> List[Dict[str, Any]]:
        """Get card groups from previous stage.

        Returns:
            List of axis card groups
        """
        reader = JSONLReader(self.input_path)
        return reader.read_as_list()

    async def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a card group for auditing.

        Args:
            item: Axis card group dict

        Returns:
            Dict with audited cards
        """
        cards = item.get("cards", [])
        if not cards:
            print(f"  No cards to audit")
            return None

        # Build prompt - use replace() to avoid conflict with JSON braces in template
        user_prompt = CARDS_AUDIT_AND_REWRITE.user
        user_prompt = user_prompt.replace("{BANNED_WORDS_JSON}", self._get_banned_words_json())
        user_prompt = user_prompt.replace("{CARDS_JSON}", json.dumps(cards, indent=2))

        # Call LLM
        content = await self.call_llm_with_retry(
            system=CARDS_AUDIT_AND_REWRITE.system,
            user=user_prompt,
        )

        if content is None:
            return None

        # Parse JSON - handle various formats
        audit_results = None
        try:
            parsed = json.loads(content)
            # Accept both list and single object
            if isinstance(parsed, list):
                audit_results = parsed
            elif isinstance(parsed, dict):
                # Single object - wrap in list
                audit_results = [parsed]
        except json.JSONDecodeError:
            # Try to find JSON array or object in content
            import re
            # Try array first
            array_match = re.search(r'\[[\s\S]*\]', content)
            if array_match:
                try:
                    audit_results = json.loads(array_match.group())
                except json.JSONDecodeError:
                    pass

            # Try object if array didn't work
            if audit_results is None:
                obj_match = re.search(r'\{[\s\S]*\}', content)
                if obj_match:
                    try:
                        parsed = json.loads(obj_match.group())
                        audit_results = [parsed] if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        pass

        if audit_results is None:
            print(f"  Could not parse JSON from response")
            print(f"  Content preview: {repr(content[:300])}...")
            return None

        # Process audit results
        audited_cards = []
        pass_count = 0
        fail_count = 0

        for audit in audit_results:
            decision = audit.get("decision", "").upper()
            card_id = audit.get("card_id")

            if decision == "PASS":
                # Find original card
                original = next(
                    (c for c in cards if c.get("card_id") == card_id),
                    None
                )
                if original:
                    audited_cards.append({
                        **original,
                        "audit_decision": "PASS",
                        "audit_reasons": audit.get("reasons", []),
                    })
                    pass_count += 1

            elif decision == "FAIL":
                fail_count += 1
                rewrite = audit.get("rewrite")
                if rewrite:
                    audited_cards.append({
                        **rewrite,
                        "audit_decision": "REWRITTEN",
                        "audit_reasons": audit.get("reasons", []),
                        "original_card_id": card_id,
                    })

        print(f"  Audit: {pass_count} pass, {fail_count} fail, {len(audited_cards)} kept")

        return {
            "id": item["id"],
            "axis_name": item["axis_name"],
            "axis_description": item.get("axis_description", ""),
            "cards": audited_cards,
            "n_cards": len(audited_cards),
            "audit_summary": {
                "passed": pass_count,
                "failed": fail_count,
                "rewritten": len(audited_cards) - pass_count,
            },
        }

    async def build_batch_requests(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build batch requests for card groups.

        Args:
            items: List of axis card groups

        Returns:
            List of batch request dicts
        """
        requests = []
        banned_json = self._get_banned_words_json()

        for item in items:
            cards = item.get("cards", [])
            if not cards:
                continue

            # Use replace() to avoid conflict with JSON braces in template
            user_prompt = CARDS_AUDIT_AND_REWRITE.user
            user_prompt = user_prompt.replace("{BANNED_WORDS_JSON}", banned_json)
            user_prompt = user_prompt.replace("{CARDS_JSON}", json.dumps(cards, indent=2))

            request = self.build_message_request(
                custom_id=item["id"],
                system=CARDS_AUDIT_AND_REWRITE.system,
                user=user_prompt,
            )
            requests.append(request)

        return requests

    async def process_batch_result(
        self,
        item: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process batch result for card audit.

        Args:
            item: Original card group
            result: Batch API result

        Returns:
            Processed output with audited cards
        """
        text = self.extract_response_text(result)
        if text is None:
            return None

        try:
            audit_results = parse_final_json(text)
        except ValueError as e:
            print(f"  Parse error: {e}")
            return None

        if audit_results is None or not isinstance(audit_results, list):
            return None

        cards = item.get("cards", [])
        audited_cards = []
        pass_count = 0
        fail_count = 0

        for audit in audit_results:
            decision = audit.get("decision", "").upper()
            card_id = audit.get("card_id")

            if decision == "PASS":
                original = next(
                    (c for c in cards if c.get("card_id") == card_id),
                    None
                )
                if original:
                    audited_cards.append({
                        **original,
                        "audit_decision": "PASS",
                        "audit_reasons": audit.get("reasons", []),
                    })
                    pass_count += 1

            elif decision == "FAIL":
                fail_count += 1
                rewrite = audit.get("rewrite")
                if rewrite:
                    audited_cards.append({
                        **rewrite,
                        "audit_decision": "REWRITTEN",
                        "audit_reasons": audit.get("reasons", []),
                        "original_card_id": card_id,
                    })

        return {
            "id": item["id"],
            "axis_name": item["axis_name"],
            "axis_description": item.get("axis_description", ""),
            "cards": audited_cards,
            "n_cards": len(audited_cards),
            "audit_summary": {
                "passed": pass_count,
                "failed": fail_count,
                "rewritten": len(audited_cards) - pass_count,
            },
        }
