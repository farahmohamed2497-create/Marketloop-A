"""
Demo: shows a real contradiction being detected and resolved by
ConsolidationLayer, with printed before/after state.

Run with:
    py mcp_server/memory/demo_conflict.py

Uses a fake LLM client (no ANTHROPIC_API_KEY needed) so it runs
standalone, same as the unit tests.
"""

import json
import re
from types import SimpleNamespace

from mcp_server.memory.episodic_store import EpisodicStore, EpisodicMemory
from mcp_server.memory.semantic_memory import SemanticMemory
from mcp_server.memory.consolidation import ConsolidationLayer


class FakeMessages:
    def create(self, model, max_tokens, system, messages):
        content = messages[0]["content"]
        entity_match = re.search(r"(customer_\w+|pet_\w+|order_\w+)", content)
        entity = entity_match.group(1) if entity_match else "unclassified"

        text_lower = content.lower()
        if "allerg" in text_lower:
            fact_type = "allergy"
        elif "delivery" in text_lower or "color" in text_lower or "prefer" in text_lower:
            fact_type = "preference"
        elif "discount" in text_lower or "promo" in text_lower:
            fact_type = "promo_offer"
        else:
            fact_type = "general_note"

        payload = json.dumps({"entity": entity, "fact_type": fact_type, "value": content})
        return SimpleNamespace(content=[SimpleNamespace(text=payload)])


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessages()


def main():
    store = EpisodicStore()
    semantic = SemanticMemory()
    consolidation = ConsolidationLayer(
        episodic_store=store,
        semantic_memory=semantic,
        llm_client=FakeAnthropicClient(),
    )

    # Two contradictory statements about the same customer, same fact_type
    store.add(EpisodicMemory(
        content="customer_412 prefers evening delivery between 6pm and 9pm",
        role="user",
        reasoning="Customer explicitly stated delivery time preference during checkout",
    ))
    store.add(EpisodicMemory(
        content="customer_412 no longer prefers evening delivery, switched to morning slots",
        role="user",
        reasoning="Customer updated delivery preference in a follow-up support call",
    ))

    print("=== BEFORE consolidation.run() ===")
    print(f"current fact for 'customer_412:preference': {semantic.get_current('customer_412:preference')}")

    consolidation.run()

    print("\n=== AFTER consolidation.run() ===")
    current = semantic.get_current("customer_412:preference")
    print(f"current fact: {current}")

    print("\n=== Full version history ===")
    for fact in semantic.get_history("customer_412:preference"):
        print(fact)

    print("\n=== Consolidation log (conflict resolution entry) ===")
    for entry in consolidation.log:
        if entry["action"] == "conflict_resolved":
            print(json.dumps(entry, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()

