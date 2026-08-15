from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ..models import EnvironmentFeedback


def deterministic_checks(goal: str, draft: str) -> list[str]:
    """Run cheap deterministic checks before LLM critique."""

    issues: list[str] = []

    if len(draft.split()) < 80:
        issues.append(
            "The deliverable is under 80 words and is probably incomplete."
        )

    goal_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z]{5,}", goal)
        if word.lower()
        not in {
            "create",
            "design",
            "write",
            "build",
            "about",
            "using",
        }
    }

    represented = [
        term
        for term in goal_terms
        if term in draft.lower()
    ]

    if goal_terms and not represented:
        issues.append(
            "The output contains none of the goal's significant terms."
        )

    if not re.search(
        r"(^|\n)(#{1,3}\s+|\d+[.)]\s+|[-*]\s+)",
        draft,
    ):
        issues.append(
            "The deliverable has no visible structure "
            "(headings or list items)."
        )

    return issues


@dataclass
class ReflectionResult:
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]


def reflect_and_refine(
    goal: str,
    draft: str,
    llm: BaseChatModel,
    *,
    grounded_check: Callable[[str], EnvironmentFeedback] | None = None,
    source_of_truth: str | None = None,
) -> ReflectionResult:
    """Critique and revise a draft using deterministic and external checks.

    ``grounded_check`` is intentionally injected by the caller.  The generic
    algorithm stays reusable, while a Sales Audit caller can provide the
    SQLite-backed evaluator that is the source of truth for factual claims.
    """

    grounded = deterministic_checks(goal, draft)

    if grounded_check is not None:
        feedback = grounded_check(draft)
        if not feedback.success:
            grounded.extend(
                f"External validation: {detail}"
                for detail in feedback.details
            )

    truth_description = source_of_truth or (
        "No external validator was supplied; only deterministic checks are available."
    )

    grounded_report = (
        "\n".join(f"- {issue}" for issue in grounded)
        or "- Deterministic checks passed."
    )

    critique_response = llm.invoke(
        [
            (
                "system",
                """You are the critique phase of a Self-Refine loop.

Judge the draft against the supplied goal and rubric.

Do not rewrite the draft.

Identify concrete correctness, completeness,
consistency, or instruction-adherence issues.

Use the deterministic checks as additional evidence.

External validator feedback is the source of truth for factual claims;
you must not answer PASS when it reports an inconsistency.""",
            ),
            (
                "human",
                f"""Goal:
{goal}

Rubric:
- correctness
- completeness
- internal consistency
- instruction adherence

SOURCE OF TRUTH:
{truth_description}

VALIDATION EVIDENCE:
{grounded_report}

Draft:
{draft}

List concrete issues.
If there are no issues, respond exactly:
PASS""",
            ),
        ],
        temperature=0.2,
    )

    critique = critique_response.content

    if not isinstance(critique, str) or not critique.strip():
        raise RuntimeError(
            "The critique phase returned an empty or unsupported response."
        )

    critique = critique.strip()

    if critique.upper() == "PASS" and not grounded:
        revised = draft
    else:
        revision_response = llm.invoke(
            [
                (
                    "system",
                    """You are the revision phase of a Self-Refine loop.

Improve the draft using:
1. external validation evidence and deterministic checks;
2. the independent critique.

Preserve correct information.
Do not invent facts or sources.
Return only the improved deliverable.""",
                ),
                (
                    "human",
                    f"""Goal:
{goal}

Original draft:
{draft}

SOURCE OF TRUTH:
{truth_description}

VALIDATION EVIDENCE:
{grounded_report}

Critique:
{critique}

Return only the improved deliverable.""",
                ),
            ],
            temperature=0.2,
        )

        revised = revision_response.content

        if not isinstance(revised, str) or not revised.strip():
            raise RuntimeError(
                "The revision phase returned an empty or unsupported response."
            )

        revised = revised.strip()

    return ReflectionResult(
        draft=draft,
        critique=critique,
        revised=revised,
        grounded_issues=grounded,
    )