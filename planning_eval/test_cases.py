"""
planning_eval/test_cases.py

Fixed benchmark suite for the real MarketLoop Sales Audit agent.

The suite uses the real SQLite database from mcp_server.db.
No mock database, fake schema, or synthetic records are created here.

The same fixed cases are reused across benchmark runs.
Do not change TEST_CASES after benchmarking starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from mcp_server.db import get_connection


# ---------------------------------------------------------------------------
# Real MarketLoop database helpers
# ---------------------------------------------------------------------------

LOW_STOCK_THRESHOLD = 20


def gt_period_totals(
    start: str,
    end: str,
) -> dict[str, float | int]:
    """Ground truth from the real Orders and Order_Items tables."""
    with get_connection() as connection:
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS order_count,
                COALESCE(SUM(total_amount), 0) AS total_revenue
            FROM Orders
            WHERE order_date BETWEEN ? AND ?
            """,
            (start, end),
        ).fetchone()

        items = connection.execute(
            """
            SELECT
                COALESCE(SUM(oi.quantity), 0) AS units_sold
            FROM Order_Items AS oi
            JOIN Orders AS o
                ON oi.order_id = o.order_id
            WHERE o.order_date BETWEEN ? AND ?
            """,
            (start, end),
        ).fetchone()

    order_count = int(totals["order_count"])
    total_revenue = round(float(totals["total_revenue"]), 2)
    units_sold = int(items["units_sold"])

    return {
        "orders": order_count,
        "revenue": total_revenue,
        "units": units_sold,
        "average_order_value": (
            round(total_revenue / order_count, 2)
            if order_count
            else 0.0
        ),
    }


def gt_low_stock_items(
    threshold: int = LOW_STOCK_THRESHOLD,
) -> list[dict[str, object]]:
    """Ground truth from the real Inventory + Products tables."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                p.product_id,
                p.product_name,
                i.quantity
            FROM Inventory AS i
            JOIN Products AS p
                ON i.product_id = p.product_id
            WHERE i.quantity <= ?
            ORDER BY i.quantity ASC, p.product_name ASC
            """,
            (threshold,),
        ).fetchall()

    return [
        {
            "product_id": int(row["product_id"]),
            "product_name": str(row["product_name"]),
            "quantity": int(row["quantity"]),
        }
        for row in rows
    ]


def gt_return_summary(
    start: str,
    end: str,
) -> dict[str, int]:
    """Ground truth from the real Return_Requests table."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_returns,
                COALESCE(
                    SUM(CASE WHEN status = 'Approved' THEN 1 ELSE 0 END),
                    0
                ) AS approved_returns,
                COALESCE(
                    SUM(CASE WHEN status = 'Rejected' THEN 1 ELSE 0 END),
                    0
                ) AS rejected_returns,
                COALESCE(
                    SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END),
                    0
                ) AS pending_returns
            FROM Return_Requests
            WHERE request_date BETWEEN ? AND ?
            """,
            (start, end),
        ).fetchone()

    return {
        "total_returns": int(row["total_returns"]),
        "approved_returns": int(row["approved_returns"]),
        "rejected_returns": int(row["rejected_returns"]),
        "pending_returns": int(row["pending_returns"]),
    }


def gt_low_stock_names() -> list[str]:
    return sorted(
        item["product_name"]
        for item in gt_low_stock_items()
    )


def gt_out_of_stock_products() -> list[str]:
    """
    Ground truth for the current inventory state.

    Note:
    The current MarketLoop Inventory table does not expose historical
    inventory snapshots, so this function checks current zero-stock items,
    not historical mid-period stock events.
    """
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT p.product_name
            FROM Inventory AS i
            JOIN Products AS p
                ON i.product_id = p.product_id
            WHERE i.quantity = 0
            ORDER BY p.product_name
            """
        ).fetchall()

    return [str(row["product_name"]) for row in rows]


def gt_pending_returns(
    start: str,
    end: str,
) -> list[dict[str, object]]:
    """Ground truth for pending return requests in a period."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                rr.return_id,
                rr.order_id,
                rr.customer_id,
                rr.reason,
                rr.request_date
            FROM Return_Requests AS rr
            WHERE rr.request_date BETWEEN ? AND ?
              AND rr.status = 'Pending'
            ORDER BY rr.request_date, rr.return_id
            """,
            (start, end),
        ).fetchall()

    return [
        {
            "return_id": int(row["return_id"]),
            "order_id": int(row["order_id"]),
            "customer_id": int(row["customer_id"]),
            "reason": str(row["reason"]),
            "request_date": str(row["request_date"]),
        }
        for row in rows
    ]


def gt_orders_for_products(
    product_ids: list[int],
    start: str,
    end: str,
) -> list[dict[str, object]]:
    """Ground truth order activity for selected products."""
    if not product_ids:
        return []

    placeholders = ",".join("?" for _ in product_ids)

    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                oi.order_id,
                oi.product_id,
                oi.quantity,
                oi.price,
                o.order_date,
                o.status,
                o.total_amount
            FROM Order_Items AS oi
            JOIN Orders AS o
                ON oi.order_id = o.order_id
            WHERE oi.product_id IN ({placeholders})
              AND o.order_date BETWEEN ? AND ?
            ORDER BY o.order_date, oi.order_id
            """,
            (*product_ids, start, end),
        ).fetchall()

    return [
        {
            "order_id": int(row["order_id"]),
            "product_id": int(row["product_id"]),
            "quantity": int(row["quantity"]),
            "price": float(row["price"]),
            "order_date": str(row["order_date"]),
            "status": str(row["status"]),
            "total_amount": float(row["total_amount"]),
        }
        for row in rows
    ]


def gt_sales_audit(
    start: str,
    end: str,
) -> dict[str, object]:
    """
    Ground truth for a complete sales-audit request.

    This intentionally uses the same real tables as the MarketLoop
    sales-audit MCP tool.
    """
    period = gt_period_totals(start, end)
    returns = gt_return_summary(start, end)
    low_stock = gt_low_stock_items()

    order_count = int(period["orders"])

    return {
        "sales": period,
        "returns": returns,
        "return_rate": (
            round(
                int(returns["total_returns"]) / order_count,
                4,
            )
            if order_count
            else 0.0
        ),
        "low_stock": low_stock,
    }


# ---------------------------------------------------------------------------
# Test case model
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    category: str
    title: str
    request: str
    params: dict = field(default_factory=dict)
    notes: str = ""
    ground_truth: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Fixed real-request benchmark suite
#
# IMPORTANT:
# Keep this list fixed once benchmark execution starts.
# ---------------------------------------------------------------------------

TEST_CASES: list[TestCase] = [

    # =======================================================================
    # A. DECOMPOSITION-FIRST
    #
    # Fixed-shape requests. The complete pipeline is known before execution.
    # =======================================================================

    TestCase(
        id="A1",
        category="DECOMP_FIRST",
        title="January complete sales audit",
        request=(
            "Run the sales audit for January 2026 "
            "(2026-01-01 to 2026-01-31). "
            "Report revenue, total orders, units sold, "
            "return activity, and current low-stock items."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "Fixed pipeline: retrieve sales, retrieve returns, "
            "retrieve inventory, then assemble the report. "
            "The required stages are known up front."
        ),
        ground_truth=gt_sales_audit,
    ),

    TestCase(
        id="A2",
        category="DECOMP_FIRST",
        title="February sales-only audit",
        request=(
            "Run a sales audit for February 2026 "
            "(2026-02-01 to 2026-02-28). "
            "Report only total revenue, total orders, and units sold."
        ),
        params={
            "start": "2026-02-01",
            "end": "2026-02-28",
        },
        notes=(
            "The request explicitly excludes inventory and returns, "
            "so the execution shape is fixed before the first step."
        ),
        ground_truth=gt_period_totals,
    ),

    TestCase(
        id="A3",
        category="DECOMP_FIRST",
        title="Q1 complete sales audit",
        request=(
            "Run the complete sales audit for Q1 2026 "
            "(2026-01-01 to 2026-03-31), including revenue, "
            "orders, units sold, return activity, and current low-stock items."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-03-31",
        },
        notes=(
            "Larger fixed scope, but the required stages remain known "
            "before execution."
        ),
        ground_truth=gt_sales_audit,
    ),

    TestCase(
        id="A4",
        category="DECOMP_FIRST",
        title="Current low-stock inventory report",
        request=(
            "List all currently low-stock products, showing the product "
            "name and current quantity, then summarize the result."
        ),
        params={},
        notes=(
            "The request is deterministic: query inventory, join product "
            "names, and format the result."
        ),
        ground_truth=gt_low_stock_items,
    ),

    TestCase(
        id="A5",
        category="DECOMP_FIRST",
        title="April returns summary",
        request=(
            "Summarize all return activity for April 2026 "
            "(2026-04-01 to 2026-04-30), including total, approved, "
            "rejected, and pending returns."
        ),
        params={
            "start": "2026-04-01",
            "end": "2026-04-30",
        },
        notes=(
            "Single known retrieval shape with no conditional branch."
        ),
        ground_truth=gt_return_summary,
    ),

    # =======================================================================
    # B. DYNAMIC
    #
    # The result of an early observation determines whether a new branch
    # should exist.
    # =======================================================================

    TestCase(
        id="B1",
        category="DYNAMIC",
        title="January audit with conditional low-stock investigation",
        request=(
            "Audit January 2026. After checking the current low-stock "
            "products, if there are any low-stock items, investigate their "
            "sales activity during January and include the findings. "
            "If there are none, skip that investigation."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "The inventory result determines whether the additional "
            "product-sales investigation branch exists."
        ),
        ground_truth=gt_low_stock_names,
    ),

    TestCase(
        id="B2",
        category="DYNAMIC",
        title="April audit with conditional pending-return investigation",
        request=(
            "Audit April 2026. If there are pending return requests, "
            "inspect the affected orders and summarize the likely issue. "
            "If there are no pending returns, skip the investigation."
        ),
        params={
            "start": "2026-04-01",
            "end": "2026-04-30",
        },
        notes=(
            "The first return-status observation determines whether the "
            "follow-up investigation should exist."
        ),
        ground_truth=gt_pending_returns,
    ),

    TestCase(
        id="B3",
        category="DYNAMIC",
        title="January audit with conditional inventory drill-down",
        request=(
            "Audit January 2026. If any product is currently out of stock, "
            "drill into its January order activity before finalizing "
            "the audit. Otherwise, skip the drill-down."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "Current zero-stock status determines whether the additional "
            "product-level order analysis is needed."
        ),
        ground_truth=gt_out_of_stock_products,
    ),

    TestCase(
        id="B4",
        category="DYNAMIC",
        title="Q1 audit with conditional return analysis",
        request=(
            "Run the Q1 2026 audit. If the period contains any pending "
            "returns, inspect those requests and their associated orders "
            "before producing the final report."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-03-31",
        },
        notes=(
            "The existence of pending returns changes the downstream "
            "execution path."
        ),
        ground_truth=gt_pending_returns,
    ),

    TestCase(
        id="B5",
        category="DYNAMIC",
        title="Low-stock investigation with conditional sales lookup",
        request=(
            "Find the current low-stock products. For each product that "
            "is low-stock, look up its sales activity for January 2026 "
            "and add that evidence to the report."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "The number and identity of the follow-up sub-tasks cannot be "
            "known until the low-stock query returns its result."
        ),
        ground_truth=gt_low_stock_items,
    ),

    # =======================================================================
    # C. LOOKAHEAD / PS / ToT / LATS
    #
    # Candidate generation and evaluation are justified by competing
    # rankings/selections.
    # =======================================================================

    TestCase(
        id="C1",
        category="LOOKAHEAD",
        title="Rank low-stock products by reorder priority",
        request=(
            "Rank the current low-stock products by reorder priority. "
            "Explain the ordering using the current inventory quantities "
            "and the reorder threshold."
        ),
        params={},
        notes=(
            "Several candidate orderings can be proposed. The ranking "
            "must be checked against real inventory values."
        ),
        ground_truth=gt_low_stock_items,
    ),

    TestCase(
        id="C2",
        category="LOOKAHEAD",
        title="Rank audit risks for management",
        request=(
            "Rank the most important operational risks in the January 2026 "
            "sales audit using evidence from sales, returns, and inventory. "
            "Explain why the top risks deserve attention first."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "Multiple plausible risk rankings exist, so candidate generation "
            "and evaluation are more appropriate than a single first answer."
        ),
        ground_truth=gt_sales_audit,
    ),

    TestCase(
        id="C3",
        category="LOOKAHEAD",
        title="Select top anomalies for escalation",
        request=(
            "From the January 2026 audit evidence, select the three findings "
            "that should be escalated to Finance first. Consider the evidence "
            "from sales, returns, and low-stock inventory."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "This is a selection problem rather than a deterministic lookup; "
            "different candidate sets can be generated and evaluated."
        ),
        ground_truth=gt_sales_audit,
    ),

    TestCase(
        id="C4",
        category="LOOKAHEAD",
        title="Choose the most actionable audit narrative",
        request=(
            "For the January 2026 sales audit, propose several possible "
            "orders for the main report sections and choose the ordering "
            "that gives management the most actionable narrative."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "The underlying facts are fixed, but multiple candidate "
            "narrative structures can be searched and compared."
        ),
        ground_truth=gt_sales_audit,
    ),

    TestCase(
        id="C5",
        category="LOOKAHEAD",
        title="Choose top low-stock escalations",
        request=(
            "Choose the three low-stock products that deserve the highest "
            "reorder attention right now. Compare multiple candidate "
            "rankings before selecting the final three."
        ),
        params={},
        notes=(
            "Requires generating and comparing multiple candidate "
            "prioritizations against real inventory evidence."
        ),
        ground_truth=gt_low_stock_items,
    ),

    # =======================================================================
    # D. REFLEXION
    #
    # Full-task retry with a bounded episodic reflection buffer.
    # =======================================================================

    TestCase(
        id="D1",
        category="REFLEXION",
        title="January audit exact database match",
        request=(
            "Produce a complete January 2026 sales audit "
            "(2026-01-01 to 2026-01-31). Every numeric claim and every "
            "low-stock claim must match the database exactly. Do not "
            "invent products or figures."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "A numeric or product hallucination in one trial should become "
            "a lesson carried into the next complete attempt."
        ),
        ground_truth=gt_sales_audit,
    ),

    TestCase(
        id="D2",
        category="REFLEXION",
        title="February exact sales reconciliation",
        request=(
            "Produce the February 2026 sales figures and make every "
            "revenue, order-count, and units-sold number match the "
            "database exactly."
        ),
        params={
            "start": "2026-02-01",
            "end": "2026-02-28",
        },
        notes=(
            "Targets repeated numeric aggregation errors across trials."
        ),
        ground_truth=gt_period_totals,
    ),

    TestCase(
        id="D3",
        category="REFLEXION",
        title="April exact return reconciliation",
        request=(
            "Produce an April 2026 returns section that exactly matches "
            "the database for total, approved, rejected, and pending "
            "return requests."
        ),
        params={
            "start": "2026-04-01",
            "end": "2026-04-30",
        },
        notes=(
            "A failed attempt can remember which return category was "
            "previously misreported and carry that lesson forward."
        ),
        ground_truth=gt_return_summary,
    ),

    TestCase(
        id="D4",
        category="REFLEXION",
        title="Exact low-stock audit",
        request=(
            "Produce the current low-stock section. Every listed product "
            "and quantity must match the Inventory and Products tables "
            "exactly, with no invented products."
        ),
        params={},
        notes=(
            "Designed to expose hallucinated products such as the iPhone "
            "case demonstrated in the grounded critique test."
        ),
        ground_truth=gt_low_stock_items,
    ),

    TestCase(
        id="D5",
        category="REFLEXION",
        title="Complete January audit with exact tie-out",
        request=(
            "Produce the complete January 2026 sales audit and ensure "
            "revenue, orders, units, returns, and low-stock findings "
            "all tie out to the database."
        ),
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        notes=(
            "Several independent factual sections must all be correct "
            "simultaneously; carrying a reflection across trials is useful "
            "when one retry fixes one section but breaks another."
        ),
        ground_truth=gt_sales_audit,
    ),
]


# ---------------------------------------------------------------------------
# Access helpers
# ---------------------------------------------------------------------------

def get_cases(
    category: Optional[str] = None,
) -> list[TestCase]:
    if category is None:
        return list(TEST_CASES)

    return [
        case
        for case in TEST_CASES
        if case.category == category
    ]


def get_case(case_id: str) -> TestCase:
    for case in TEST_CASES:
        if case.id == case_id:
            return case

    raise KeyError(f"Unknown test case: {case_id}")


if __name__ == "__main__":
    from collections import Counter

    counts = Counter(case.category for case in TEST_CASES)

    print(f"Total cases: {len(TEST_CASES)}")

    for category in (
        "DECOMP_FIRST",
        "DYNAMIC",
        "LOOKAHEAD",
        "REFLEXION",
    ):
        print(f"{category}: {counts.get(category, 0)}")