"""Deliberately weak, non-database baseline for the LATS comparison.

It exists only to demonstrate why a format/self-critique signal is weaker
than the SQLite-backed evaluator. It must never be used as the shipping
environment or labeled grounded.
"""

from __future__ import annotations

from planning_lab.models import EnvironmentFeedback


class UngroundedFormatEnvironment:
    """Accept a readable audit-shaped answer without checking any facts."""

    grounded = False
    source_of_truth = "No external source of truth; format-only baseline."
    candidate_contract = """Write a complete, readable audit response with at least
one heading and references to sales, returns, or inventory. This baseline does
not validate factual correctness and is used only for the comparison."""

    def evaluate(self, state: str) -> EnvironmentFeedback:
        text = state.strip() if isinstance(state, str) else ""
        has_structure = "#" in text or "\n-" in text or "\n1." in text
        has_audit_term = any(
            term in text.lower()
            for term in ("sales", "returns", "inventory", "low-stock", "audit")
        )
        success = len(text.split()) >= 20 and has_structure and has_audit_term
        return EnvironmentFeedback(
            success=success,
            score=1.0 if success else 0.0,
            details=(
                ["Format-only baseline accepted the candidate without factual verification."]
                if success
                else ["Candidate needs a structured audit-shaped response of at least 20 words."]
            ),
        )