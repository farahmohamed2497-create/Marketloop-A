"""
Consolidation layer.
Runs periodically, separate from the live agent loop and separate from
the promote-or-drop router. Reads promoted episodes from EpisodicStore
and writes versioned facts into SemanticMemory.
This is the ONLY module allowed to call SemanticMemory.write_fact().
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import anthropic

from mcp_server.memory.episodic_store import EpisodicStore, EpisodicMemory
from mcp_server.memory.semantic_memory import SemanticMemory


EXTRACTION_SYSTEM_PROMPT = """You extract structured facts from a memory note.
Return ONLY a JSON object, no markdown, no preamble, in this exact shape:
{"entity": "...", "fact_type": "...", "value": "..."}

- entity: the customer/pet/order/product identifier mentioned in the text
  (e.g. "customer_412", "pet_7"). If no clear identifier exists, use "unclassified".
- fact_type: a short snake_case category label that best fits this fact
  (e.g. allergy, preference, delivery_note, payment_issue, complaint,
  stock_note, return_request). Infer the best-fitting label yourself,
  do not restrict to a fixed list.
- value: keep the original text unchanged.
"""


class ConsolidationLayer:
    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_memory: SemanticMemory,
        llm_client: anthropic.Anthropic | None = None,
    ):
        self.episodic_store = episodic_store
        self.semantic_memory = semantic_memory
        self.log: list[dict] = []
        self.client = llm_client or anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def run(self) -> None:
       
        episodes = self.episodic_store.get_all()
        extracted = [self._extract(ep) for ep in episodes]
        extracted = [e for e in extracted if e is not None]

        grouped = self._group_by_key(extracted)

        for key, items in grouped.items():
            self._consolidate_group(key, items)

        self._expire_stale_facts()

    def _extract(self, episode: EpisodicMemory) -> dict | None:
       
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": episode.content}],
            )
            raw_text = response.content[0].text.strip()
            raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            parsed = json.loads(raw_text)
        except (json.JSONDecodeError, IndexError, anthropic.APIError) as exc:
           
            self._log(
                key="extraction_error",
                action="extract_failed",
                old_value=None,
                new_value=episode.content,
                error=str(exc),
            )
            return None

        entity = parsed.get("entity") or "unclassified"
        fact_type = parsed.get("fact_type") or "general_note"
        key = f"{entity}:{fact_type}"

        return {
            "key": key,
            "value": episode.content,
            "timestamp": episode.timestamp,
        }

    def _group_by_key(self, items: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            grouped[item["key"]].append(item)
        return grouped

    def _consolidate_group(self, key: str, items: list[dict]) -> None:
      items.sort(key=lambda i: i["timestamp"])

      for item in items:
        current = self.semantic_memory.get_current(key)

        if current is not None and current.value == item["value"]:
            continue

        new_fact = self.semantic_memory.write_fact(
            key=key,
            value=item["value"],
            source_episodes=[item["timestamp"]],
        )

        if current is None:
            self._log(key, "created", None, new_fact.value)
        else:
            self._log(
                key, "conflict_resolved", current.value, new_fact.value,
                old_fact_id=current.id, new_fact_id=new_fact.id,
            )

    def _resolve_conflict(self, key: str, current_fact, latest_item: dict) -> None:
        
        new_fact = self.semantic_memory.write_fact(
            key=key,
            value=latest_item["value"],
            source_episodes=[latest_item["timestamp"]],
        )

        self._log(
            key,
            "conflict_resolved",
            current_fact.value,
            new_fact.value,
            old_fact_id=current_fact.id,
            new_fact_id=new_fact.id,
        )

    def _expire_stale_facts(self) -> None:
       
        expired = self.semantic_memory.expire_stale_facts()
        for fact in expired:
            self._log(fact.key, "expired", fact.value, None, fact_id=fact.id)

    def _log(self, key: str, action: str, old_value, new_value, **extra) -> None:
        self.log.append({
            "timestamp": time.time(),
            "key": key,
            "action": action,
            "old_value": old_value,
            "new_value": new_value,
            **extra,
        })