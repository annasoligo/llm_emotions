"""Stage 4: Audit scenario pairs for quality."""

import json
from typing import Any, Dict, List, Optional

from probes.data.appraisal_prompt import SCENARIO_PAIR_AUDIT
from probes.scripts.appraisal.stages.base import Stage
from probes.scripts.appraisal.utils.parsing import parse_final_json
from probes.scripts.appraisal.utils.validation import (
    get_combined_forbidden_words,
    validate_scenario_pair,
)
from probes.scripts.appraisal.utils.jsonl_writer import JSONLReader


class ScenarioAuditStage(Stage):
    """Audit scenario pairs for minimality and forbidden words.

    Two-phase audit:
    1. Heuristic checks (local validation)
    2. LLM audit for semantic quality

    Input: scenarios.jsonl
    Output: scenarios_audited.jsonl with audit results
    """

    name = "scenario_audit"
    input_file = "scenarios.jsonl"
    output_file = "scenarios_audited.jsonl"

    def get_input_items(self) -> List[Dict[str, Any]]:
        """Get scenarios from previous stage.

        Returns:
            List of scenario pair dicts
        """
        reader = JSONLReader(self.input_path)
        return reader.read_as_list()

    def _get_forbidden_set(self, card: Dict[str, Any]) -> set:
        """Get forbidden words set for a card."""
        card_forbidden = card.get("forbidden_in_scenarios", [])
        return get_combined_forbidden_words(
            card_forbidden=card_forbidden,
            additional_forbidden=self.config.additional_forbidden_words,
        )

    def _run_heuristic_audit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run local heuristic checks on scenario pair.

        Args:
            item: Scenario pair dict

        Returns:
            Validation result dict
        """
        scenario_a = item.get("scenario_a", "")
        scenario_b = item.get("scenario_b", "")
        card = item.get("card", {})

        forbidden = self._get_forbidden_set(card)

        return validate_scenario_pair(
            text_a=scenario_a,
            text_b=scenario_b,
            forbidden_words=forbidden,
            max_diff_ratio=self.config.max_diff_ratio,
        )

    async def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a scenario pair for auditing.

        Args:
            item: Scenario pair dict

        Returns:
            Dict with audit results
        """
        # Phase 1: Heuristic audit
        heuristic_result = self._run_heuristic_audit(item)

        # Phase 2: LLM audit
        card = item.get("card", {})
        forbidden_list = list(self._get_forbidden_set(card))

        # Use replace() to avoid conflict with JSON braces in template
        user_prompt = SCENARIO_PAIR_AUDIT.user
        user_prompt = user_prompt.replace("{CARD_JSON}", json.dumps(card, indent=2))
        user_prompt = user_prompt.replace("{A_TEXT}", item["scenario_a"])
        user_prompt = user_prompt.replace("{B_TEXT}", item["scenario_b"])
        user_prompt = user_prompt.replace("{FORBIDDEN_LIST_JSON}", json.dumps(forbidden_list))

        content = await self.call_llm_with_retry(
            system=SCENARIO_PAIR_AUDIT.system,
            user=user_prompt,
        )

        llm_result = None
        if content:
            try:
                llm_result = json.loads(content)
            except json.JSONDecodeError:
                pass

        # Combine results
        combined_pass = heuristic_result["pass"]
        if llm_result:
            combined_pass = combined_pass and llm_result.get("pass", False)

        return {
            "id": item["id"],
            "axis_name": item.get("axis_name"),
            "card_id": item.get("card_id"),
            "domain": item.get("domain"),
            "scenario_idx": item.get("scenario_idx"),
            "scenario_a": item["scenario_a"],
            "scenario_b": item["scenario_b"],
            "changed_fact": item.get("changed_fact"),
            "card": card,
            "audit_pass": combined_pass,
            "heuristic_audit": heuristic_result,
            "llm_audit": llm_result,
        }

    async def build_batch_requests(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build batch requests for scenario audits.

        Args:
            items: List of scenario pairs

        Returns:
            List of batch request dicts
        """
        requests = []

        for item in items:
            card = item.get("card", {})
            forbidden_list = list(self._get_forbidden_set(card))

            # Use replace() to avoid conflict with JSON braces in template
            user_prompt = SCENARIO_PAIR_AUDIT.user
            user_prompt = user_prompt.replace("{CARD_JSON}", json.dumps(card, indent=2))
            user_prompt = user_prompt.replace("{A_TEXT}", item["scenario_a"])
            user_prompt = user_prompt.replace("{B_TEXT}", item["scenario_b"])
            user_prompt = user_prompt.replace("{FORBIDDEN_LIST_JSON}", json.dumps(forbidden_list))

            request = self.build_message_request(
                custom_id=item["id"],
                system=SCENARIO_PAIR_AUDIT.system,
                user=user_prompt,
            )
            requests.append(request)

        return requests

    async def process_batch_result(
        self,
        item: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process batch result for scenario audit.

        Args:
            item: Original scenario pair
            result: Batch API result

        Returns:
            Processed audit result
        """
        # Phase 1: Heuristic audit (always run)
        heuristic_result = self._run_heuristic_audit(item)

        # Phase 2: LLM audit
        text = self.extract_response_text(result)
        llm_result = None

        if text:
            try:
                llm_result = parse_final_json(text)
            except ValueError:
                pass

        # Combine results
        combined_pass = heuristic_result["pass"]
        if llm_result:
            combined_pass = combined_pass and llm_result.get("pass", False)

        card = item.get("card", {})

        return {
            "id": item["id"],
            "axis_name": item.get("axis_name"),
            "card_id": item.get("card_id"),
            "domain": item.get("domain"),
            "scenario_idx": item.get("scenario_idx"),
            "scenario_a": item["scenario_a"],
            "scenario_b": item["scenario_b"],
            "changed_fact": item.get("changed_fact"),
            "card": card,
            "audit_pass": combined_pass,
            "heuristic_audit": heuristic_result,
            "llm_audit": llm_result,
        }
