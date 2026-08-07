"""
Unit tests for the ConsolidationLayer + SemanticMemory.
Covers: creation, updates (change over time), conflict resolution,
and expiration -- run independently from the live agent loop.

Uses a fake LLM client (no real API calls) so tests stay fast,
deterministic, and don't require ANTHROPIC_API_KEY.
"""

import json
import re
import time
from types import SimpleNamespace

import pytest

from mcp_server.memory.episodic_store import EpisodicStore, EpisodicMemory
from mcp_server.memory.semantic_memory import SemanticMemory
from mcp_server.memory.consolidation import ConsolidationLayer


# ---------- Fake LLM client (test double, no real API calls) ----------

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

        payload = json.dumps({
            "entity": entity,
            "fact_type": fact_type,
            "value": content,
        })

        return SimpleNamespace(content=[SimpleNamespace(text=payload)])


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessages()


# ---------- Helpers ----------

def _make_layer():
    store = EpisodicStore()
    semantic = SemanticMemory()
    consolidation = ConsolidationLayer(
        episodic_store=store,
        semantic_memory=semantic,
        llm_client=FakeAnthropicClient(),
    )
    return store, semantic, consolidation


# ---------- Creation ----------

def test_first_episode_creates_semantic_fact():
    store, semantic, consolidation = _make_layer()

    store.add(EpisodicMemory(
        content="customer_101 has peanut allergy",
        role="user",
        reasoning="stated during onboarding",
    ))

    consolidation.run()

    current = semantic.get_current("customer_101:allergy")
    assert current is not None
    assert "peanut" in current.value
    assert current.version == 1

    created_logs = [e for e in consolidation.log if e["action"] == "created"]
    assert len(created_logs) == 1
    assert created_logs[0]["old_value"] is None


# ---------- Update  ----------

def test_consolidation_updates_fact_when_value_changes():
    store, semantic, consolidation = _make_layer()

    store.add(EpisodicMemory(
        content="customer_205 prefers evening delivery between 6pm and 9pm",
        role="user",
        reasoning="stated at checkout",
    ))
    store.add(EpisodicMemory(
        content="customer_205 no longer prefers evening delivery, switched to morning slots",
        role="user",
        reasoning="updated via support call",
    ))

    consolidation.run()

    current = semantic.get_current("customer_205:preference")
    assert "morning" in current.value
    assert current.version == 2

    history = semantic.get_history("customer_205:preference")
    assert len(history) == 2
    assert history[0].superseded_by == history[1].id 

def test_no_new_write_when_value_is_identical():
    store, semantic, consolidation = _make_layer()

    store.add(EpisodicMemory(
        content="customer_301 prefers blue color",
        role="user", reasoning="stated once",
    ))
    store.add(EpisodicMemory(
        content="customer_301 prefers blue color",  
        role="user", reasoning="repeated in a later call",
    ))

    consolidation.run()

    history = semantic.get_history("customer_301:preference")
    assert len(history) == 1  


# ---------- Conflict resolution ----------

def test_consolidation_resolves_real_conflict():
   
    store, semantic, consolidation = _make_layer()

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

    consolidation.run()

    current = semantic.get_current("customer_412:preference")
    assert "morning" in current.value
    assert current.version == 2

    conflict_logs = [e for e in consolidation.log if e["action"] == "conflict_resolved"]
    assert len(conflict_logs) == 1
    assert conflict_logs[0]["old_value"] == "customer_412 prefers evening delivery between 6pm and 9pm"
    assert "morning" in conflict_logs[0]["new_value"]

    
    history = semantic.get_history("customer_412:preference")
    assert len(history) == 2
    assert history[0].superseded_by is not None


# ---------- Expiration ----------

def test_expire_stale_facts_marks_expired_facts():
    semantic = SemanticMemory()

    semantic.write_fact(
        key="customer_501:promo_offer",
        value="10% discount valid until end of campaign",
        source_episodes=["ep_1"],
        valid_until=time.time() - 1, 
    )

    expired = semantic.expire_stale_facts()

    assert len(expired) == 1
    assert expired[0].key == "customer_501:promo_offer"


def test_expire_stale_facts_ignores_facts_without_expiry():
    semantic = SemanticMemory()

    semantic.write_fact(
        key="customer_502:allergy",
        value="peanut allergy",
        source_episodes=["ep_1"],
        valid_until=None,  
    )

    expired = semantic.expire_stale_facts()

    assert len(expired) == 0


def test_consolidation_logs_expired_facts():
    store, semantic, consolidation = _make_layer()

   
    semantic.write_fact(
        key="customer_601:promo_offer",
        value="15% discount",
        source_episodes=["ep_old"],
        valid_until=time.time() - 100,
    )

    consolidation.run() 

    expired_logs = [e for e in consolidation.log if e["action"] == "expired"]
    assert len(expired_logs) == 1
    assert expired_logs[0]["key"] == "customer_601:promo_offer"


# ---------- Extraction failure handling ----------

def test_extraction_failure_is_logged_not_crashed():
    
    class BrokenMessages:
        def create(self, model, max_tokens, system, messages):
            return SimpleNamespace(content=[SimpleNamespace(text="not valid json at all")])

    class BrokenClient:
        def __init__(self):
            self.messages = BrokenMessages()

    store = EpisodicStore()
    semantic = SemanticMemory()
    consolidation = ConsolidationLayer(
        episodic_store=store,
        semantic_memory=semantic,
        llm_client=BrokenClient(),
    )

    store.add(EpisodicMemory(
        content="customer_701 has some note",
        role="user",
        reasoning="test",
    ))

    consolidation.run() 

    failed_logs = [e for e in consolidation.log if e["action"] == "extract_failed"]
    assert len(failed_logs) == 1