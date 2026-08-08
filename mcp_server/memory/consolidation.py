"""Periodic, auditable promotion from episodic to semantic memory."""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Any

import anthropic

from mcp_server.memory.episodic_store import EpisodicMemory, EpisodicStore
from mcp_server.memory.semantic_memory import SemanticMemory


EXTRACTION_SYSTEM_PROMPT = """Extract one memory fact. Return JSON only:
{"entity": "...", "fact_type": "...", "value": "..."}
Use a stable identifier such as customer_123 when present. Keep value faithful
to the source note and use a short snake_case fact_type."""

# A fact may have no expiry when it is a durable preference or safety fact.
TTL_SECONDS = {
    "promo_offer": 30 * 24 * 60 * 60,
    "return_request": 90 * 24 * 60 * 60,
    "payment_issue": 90 * 24 * 60 * 60,
    "delivery_note": 180 * 24 * 60 * 60,
    "complaint": 180 * 24 * 60 * 60,
}


class ConsolidationLayer:
    """Runs separately from the live loop; it alone writes semantic facts."""

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_memory: SemanticMemory,
        llm_client: anthropic.Anthropic | None = None,
    ) -> None:
        self.episodic_store = episodic_store
        self.semantic_memory = semantic_memory
        self.log: list[dict[str, Any]] = []
        # Do not construct an API client when no key is configured. This makes
        # local demos and the agent usable without an external dependency.
        self.client = llm_client
        if self.client is None and os.getenv("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def run(self) -> None:
        extracted = [self._extract(episode) for episode in self.episodic_store.get_all()]
        grouped = self._group_by_key(item for item in extracted if item is not None)
        for key, items in grouped.items():
            self._consolidate_group(key, items)
        self._expire_stale_facts()

    def _extract(self, episode: EpisodicMemory) -> dict[str, Any] | None:
        if self.client is None:
            return self._rule_based_extract(episode)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": episode.content}],
            )
            raw = response.content[0].text.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
        except (json.JSONDecodeError, IndexError, AttributeError, anthropic.APIError) as exc:
            self._log("extraction_error", "extract_failed", None, episode.content, error=str(exc))
            return None

        return self._normalise_extraction(parsed, episode)

    def _rule_based_extract(self, episode: EpisodicMemory) -> dict[str, Any]:
        """Small offline fallback; the original note remains the source of truth."""
        content = episode.content
        # Matches "order_1", "order #1", "order # 1", and "customer 123" alike,
        # then normalises to the canonical "<type>_<id>" key form. The id group
        # requires a digit so a plain sentence subject like "Customer wants..."
        # doesn't false-match as an identifier.
        entity_match = re.search(
            r"(customer|pet|order|product)[\s_]*#?\s*(\d+)", content, re.IGNORECASE
        )
        entity = (
            f"{entity_match.group(1).lower()}_{entity_match.group(2)}"
            if entity_match
            else "unclassified"
        )
        lowered = content.lower()
        if "allerg" in lowered:
            fact_type = "allergy"
        elif "promo" in lowered or "discount" in lowered:
            fact_type = "promo_offer"
        elif "return" in lowered or "damaged" in lowered:
            fact_type = "return_request"
        elif "payment" in lowered or "refund" in lowered:
            fact_type = "payment_issue"
        elif "complaint" in lowered:
            fact_type = "complaint"
        elif "delivery" in lowered or "prefer" in lowered or "color" in lowered:
            fact_type = "preference"
        else:
            fact_type = "general_note"
        return self._normalise_extraction({"entity": entity, "fact_type": fact_type}, episode)

    def _normalise_extraction(self, parsed: dict[str, Any], episode: EpisodicMemory) -> dict[str, Any]:
        entity = str(parsed.get("entity") or "unclassified")
        fact_type = str(parsed.get("fact_type") or "general_note")
        return {
            "key": f"{entity}:{fact_type}",
            # Always retain the original episode; an extractor is a classifier,
            # not an authority allowed to rewrite customer facts.
            "value": episode.content,
            "timestamp": episode.timestamp,
            "valid_until": self._valid_until(fact_type),
        }

    @staticmethod
    def _valid_until(fact_type: str) -> float | None:
        ttl = TTL_SECONDS.get(fact_type)
        return time.time() + ttl if ttl is not None else None

    @staticmethod
    def _group_by_key(items: Any) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[item["key"]].append(item)
        return grouped

    def _consolidate_group(self, key: str, items: list[dict[str, Any]]) -> None:
        for item in sorted(items, key=lambda value: value["timestamp"]):
            current = self.semantic_memory.get_current(key)
            if current is not None and current.value == item["value"]:
                continue
            new_fact = self.semantic_memory.write_fact(
                key=key,
                value=item["value"],
                source_episodes=[item["timestamp"]],
                valid_until=item["valid_until"],
            )
            if current is None:
                self._log(key, "created", None, new_fact.value, new_fact_id=new_fact.id)
            else:
                self._log(
                    key, "conflict_resolved", current.value, new_fact.value,
                    old_fact_id=current.id, new_fact_id=new_fact.id,
                )

    def _expire_stale_facts(self) -> None:
        for fact in self.semantic_memory.expire_stale_facts():
            self._log(fact.key, "expired", fact.value, None, fact_id=fact.id)

    def _log(self, key: str, action: str, old_value: Any, new_value: Any, **extra: Any) -> None:
        self.log.append({
            "timestamp": time.time(), "key": key, "action": action,
            "old_value": old_value, "new_value": new_value, **extra,
        })