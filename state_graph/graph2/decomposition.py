"""LLM task-decomposition node for shipping investigations."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel


class ShippingTaskDecomposer:
    """Turn a free-text shipping complaint into durable executable steps."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def decompose(self, goal: str) -> list[str]:
        if not goal.strip():
            raise ValueError("A shipping issue is required for decomposition.")

        response = self.llm.invoke(
            "Break the following shipping issue into 2-5 ordered, concrete "
            "subtasks. Use only tracking checks and carrier-claim actions. "
            "Return one subtask per line without numbering.\n\n"
            f"Issue: {goal}"
        )
        content = getattr(response, "content", "")

        if not isinstance(content, str):
            raise ValueError("The decomposition model returned non-text content.")

        subtasks = [
            line.strip().lstrip("-0123456789. ").strip()
            for line in content.splitlines()
            if line.strip()
        ]

        if not 2 <= len(subtasks) <= 5 or any(not task for task in subtasks):
            raise ValueError("Expected 2-5 non-empty shipping subtasks.")

        return subtasks
