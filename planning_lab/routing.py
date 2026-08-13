from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.models import EnvironmentFeedback
from mcp_server.db import get_connection
from planning_lab.algorithms.environment import Environment
from planning_lab.algorithms.reflexion import ReflexionResult, reflexion


class PlanningMethod(StrEnum):
    SELF_REFINE = "self_refine"
    REFLEXION = "reflexion"


@dataclass(frozen=True)
class SubtaskProfile:
    cheap_to_redo: bool
    needs_cross_trial_learning: bool


@dataclass
class GroundedEvidence:
    passed: bool
    source_of_truth: str
    evidence: list[str]


@dataclass
class SelfRefineResult:
    draft: str
    critique: str
    revised: str
    grounded_evidence: GroundedEvidence


@dataclass
class SelfCorrectionResult:
    method: PlanningMethod
    success: bool
    output: str
    attempts: int
    source_of_truth: str
    self_refine_result: SelfRefineResult | None = None
    reflexion_result: ReflexionResult | None = None


def classify_subtask(
    task: str,
    *,
    cheap_to_redo: bool | None = None,
    needs_cross_trial_learning: bool | None = None,
) -> SubtaskProfile:
    """
    Decide whether a sub-task is better handled by one refinement cycle
    or repeated trials with episodic learning.

    Explicit flags override inference.
    """
    normalized = task.strip().lower()

    learning_signals = (
        "debug",
        "troubleshoot",
        "retry",
        "recover",
        "root cause",
        "multiple attempts",
        "multiple trials",
        "iteratively",
        "persistent failure",
        "fix until",
    )

    cheap_signals = (
        "rewrite",
        "edit",
        "polish",
        "improve",
        "refine",
        "summarize",
        "format",
        "review",
        "draft",
        "clarify",
    )

    if needs_cross_trial_learning is None:
        needs_cross_trial_learning = any(
            signal in normalized
            for signal in learning_signals
        )

    if cheap_to_redo is None:
        cheap_to_redo = any(
            signal in normalized
            for signal in cheap_signals
        ) or not needs_cross_trial_learning

    return SubtaskProfile(
        cheap_to_redo=cheap_to_redo,
        needs_cross_trial_learning=needs_cross_trial_learning,
    )


def route_self_correction(profile: SubtaskProfile) -> PlanningMethod:
    if profile.needs_cross_trial_learning:
        return PlanningMethod.REFLEXION

    return PlanningMethod.SELF_REFINE


# ---------------------------------------------------------------------------
# Grounded validation
# ---------------------------------------------------------------------------

def _extract_date_range(task: str) -> tuple[str, str] | None:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", task)

    if len(dates) >= 2:
        return dates[0], dates[1]

    return None


def _run_sales_grounding(
    task: str,
    draft: str,
) -> GroundedEvidence:
    """
    Validate sales-audit claims against the real SQLite database.

    Source of truth:
    - Orders table
    - Order_Items table
    - Inventory table
    - Products table
    """
    date_range = _extract_date_range(task)

    if date_range is None:
        return GroundedEvidence(
            passed=True,
            source_of_truth=(
                "No database-backed sales-audit validator was applicable "
                "because no date range was found in the task."
            ),
            evidence=[],
        )

    start_date, end_date = date_range

    with get_connection() as connection:
        # Sales totals
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS order_count,
                COALESCE(SUM(total_amount), 0) AS total_revenue
            FROM Orders
            WHERE order_date BETWEEN ? AND ?
            """,
            (start_date, end_date),
        ).fetchone()

        # Units sold
        items = connection.execute(
            """
            SELECT
                COALESCE(SUM(oi.quantity), 0) AS units_sold
            FROM Order_Items AS oi
            JOIN Orders AS o
                ON oi.order_id = o.order_id
            WHERE o.order_date BETWEEN ? AND ?
            """,
            (start_date, end_date),
        ).fetchone()

        # Current low-stock inventory
        low_stock_rows = connection.execute(
            """
            SELECT
                p.product_name,
                i.quantity
            FROM Inventory AS i
            JOIN Products AS p
                ON i.product_id = p.product_id
            WHERE i.quantity <= 20
            ORDER BY i.quantity ASC
            """
        ).fetchall()

    actual_orders = int(totals["order_count"])
    actual_revenue = round(float(totals["total_revenue"]), 2)
    actual_units = int(items["units_sold"])

    actual_low_stock = {
        row["product_name"].strip().lower(): int(row["quantity"])
        for row in low_stock_rows
    }

    evidence: list[str] = []
    draft_lower = draft.lower().replace(",", "")

    # ------------------------------------------------------------
    # Sales metrics
    # ------------------------------------------------------------

    expected_claims = {
        "total orders": str(actual_orders),
        "total revenue": f"${actual_revenue:,.2f}",
        "units sold": str(actual_units),
    }

    for label, expected in expected_claims.items():
        expected_normalized = expected.lower().replace(",", "")

        if label not in draft_lower:
            evidence.append(
                f"Draft does not explicitly state {label}; "
                f"database value is {expected}."
            )
            continue

        if expected_normalized not in draft_lower:
            evidence.append(
                f"Mismatch for {label}: database value is {expected}."
            )

    # ------------------------------------------------------------
    # Low-stock inventory
    # ------------------------------------------------------------

    low_stock_match = re.search(
        r"low[- ]stock items\s*:\s*(.*?)(?=\n\s*\n|\Z)",
        draft,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if low_stock_match:
        low_stock_text = low_stock_match.group(1)

        draft_low_stock: dict[str, int] = {}

        for product, quantity in re.findall(
            r"-\s*(.+?):\s*(\d+)\s*units?",
            low_stock_text,
            flags=re.IGNORECASE,
        ):
            draft_low_stock[product.strip().lower()] = int(quantity)

        # Check every item claimed by the draft.
        for product, claimed_quantity in draft_low_stock.items():
            actual_quantity = actual_low_stock.get(product)

            if actual_quantity is None:
                evidence.append(
                    f"Mismatch for low-stock item '{product}': "
                    "the product is not present in the database low-stock results."
                )
            elif actual_quantity != claimed_quantity:
                evidence.append(
                    f"Mismatch for low-stock item '{product}': "
                    f"draft says {claimed_quantity} units, "
                    f"database says {actual_quantity} units."
                )

        # Check for database items missing from the draft.
        for product, actual_quantity in actual_low_stock.items():
            if product not in draft_low_stock:
                evidence.append(
                    f"Draft is missing database low-stock item "
                    f"'{product}' ({actual_quantity} units)."
                )

    passed = not evidence

    return GroundedEvidence(
        passed=passed,
        source_of_truth=(
            "SQLite database: Orders, Order_Items, Inventory, and Products "
            f"tables for {start_date} to {end_date}; "
            "inventory low-stock threshold = 20 units."
        ),
        evidence=evidence,
    )

# ---------------------------------------------------------------------------
# Self-Refine
# ---------------------------------------------------------------------------

def _run_self_refine(
    task: str,
    llm: BaseChatModel,
) -> SelfRefineResult:
    draft_response = llm.invoke(
        [
            (
                "system",
                "Produce the complete deliverable for the task.",
            ),
            (
                "human",
                f"""Task:
{task}

Return only the complete draft.""",
            ),
        ],
        temperature=0.2,
    )

    draft = draft_response.content

    if not isinstance(draft, str) or not draft.strip():
        raise RuntimeError(
            "The draft phase returned an empty or unsupported response."
        )

    draft = draft.strip()

    grounded = _run_sales_grounding(
        task=task,
        draft=draft,
    )

    evidence_text = (
        "\n".join(f"- {item}" for item in grounded.evidence)
        if grounded.evidence
        else "- No grounded inconsistencies found."
    )

    critique_response = llm.invoke(
        [
            (
                "system",
                """You are an independent critique stage in a Self-Refine loop.

Evaluate the draft against the task and the explicit rubric.

Grounded evidence is external evidence and must be treated as the
source of truth when checking factual claims.

Do not rewrite the answer.""",
            ),
            (
                "human",
                f"""Task:
{task}

Rubric:
- correctness
- completeness
- internal consistency
- instruction adherence

SOURCE OF TRUTH:
{grounded.source_of_truth}

GROUNDED EVIDENCE:
{evidence_text}

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

    if critique.upper() == "PASS" and grounded.passed:
        revised = draft
    else:
        revision_response = llm.invoke(
            [
                (
                    "system",
                    """You are the revision stage of Self-Refine.

Revise the draft using the grounded evidence and critique.
Preserve correct information.
Never contradict the source of truth.
Return only the revised deliverable.""",
                ),
                (
                    "human",
                    f"""Task:
{task}

Original draft:
{draft}

SOURCE OF TRUTH:
{grounded.source_of_truth}

GROUNDED EVIDENCE:
{evidence_text}

Critique:
{critique}

Return only the revised deliverable.""",
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

    return SelfRefineResult(
        draft=draft,
        critique=critique,
        revised=revised,
        grounded_evidence=grounded,
    )


# ---------------------------------------------------------------------------
# Public self-correction entry point
# ---------------------------------------------------------------------------

def self_correct(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    *,
    profile: SubtaskProfile | None = None,
    cheap_to_redo: bool | None = None,
    needs_cross_trial_learning: bool | None = None,
    reflexion_trials: int = 3,
    reflexion_memory_size: int = 3,
) -> SelfCorrectionResult:

    if profile is None:
        profile = classify_subtask(
            task,
            cheap_to_redo=cheap_to_redo,
            needs_cross_trial_learning=needs_cross_trial_learning,
        )

    method = route_self_correction(profile)

    if method == PlanningMethod.SELF_REFINE:
        result = _run_self_refine(
            task=task,
            llm=llm,
        )

        return SelfCorrectionResult(
            method=method,
            success=bool(result.revised.strip()),
            output=result.revised,
            attempts=1,
            source_of_truth=result.grounded_evidence.source_of_truth,
            self_refine_result=result,
        )

    reflexion_result = reflexion(
        task=task,
        llm=llm,
        environment=environment,
        max_trials=reflexion_trials,
        memory_size=reflexion_memory_size,
    )

    return SelfCorrectionResult(
        method=method,
        success=reflexion_result.success,
        output=reflexion_result.output,
        attempts=len(reflexion_result.trials),
        source_of_truth=(
            "EnvironmentFeedback from environment.evaluate(); "
            "the current Environment is stochastic and will be replaced "
            "by a real external validator later."
        ),
        reflexion_result=reflexion_result,
    )