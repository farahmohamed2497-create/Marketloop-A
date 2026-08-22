from types import SimpleNamespace

import pytest

from state_graph.graph2.decomposition import ShippingTaskDecomposer


class _LLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self.content)


def test_shipping_task_decomposer_normalizes_and_limits_subtasks():
    decomposer = ShippingTaskDecomposer(
        _LLM("1. Check tracking\n- Verify delivery address\n3. Open carrier claim")
    )

    assert decomposer.decompose("My package is missing.") == [
        "Check tracking",
        "Verify delivery address",
        "Open carrier claim",
    ]


def test_shipping_task_decomposer_rejects_invalid_llm_output():
    decomposer = ShippingTaskDecomposer(_LLM("Only one step"))

    with pytest.raises(ValueError, match="2-5"):
        decomposer.decompose("My package is missing.")
