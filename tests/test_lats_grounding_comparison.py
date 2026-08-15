"""Regression proof that the ungrounded baseline can accept an invented answer."""

from __future__ import annotations

from planning_eval.grounded_environment import CaseGroundedEnvironment
from planning_eval.test_cases import get_case
from planning_eval.ungrounded_environment import UngroundedFormatEnvironment


def test_grounded_and_ungrounded_environments_diverge_on_invented_audit() -> None:
    invented = """# January Audit

- Sales were excellent and revenue was 999999.00.
- Returns are fully resolved with no operational concern.
- Inventory is healthy, so no low-stock follow-up is needed.
"""

    assert UngroundedFormatEnvironment().evaluate(invented).success is True
    assert CaseGroundedEnvironment(get_case("A1")).evaluate(invented).success is False