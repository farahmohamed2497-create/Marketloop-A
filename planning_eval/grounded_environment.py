"""Database-backed benchmark evaluator for the fixed Sales Audit suite.

This module deliberately does not use an LLM or a random score.  It checks
claims in a candidate response against the fixed case's ground-truth function,
which queries the MarketLoop SQLite database.
"""

from __future__ import annotations

import re
import inspect
from collections.abc import Iterator
from typing import Any

from planning_lab.models import EnvironmentFeedback

from .test_cases import TestCase


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace(",", " ")).strip()


def _number_variants(value: int | float) -> set[str]:
    """Return safe textual forms that represent one database number."""
    if isinstance(value, int) or float(value).is_integer():
        integer = int(value)
        return {str(integer), f"{integer:,}".replace(",", " ")}
    return {f"{value:.2f}", f"{value:,.2f}".replace(",", " ")}


def _walk_truth(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_truth(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_truth(item, (*path, str(index)))
    else:
        yield path, value


class CaseGroundedEnvironment:
    """Evaluate one benchmark response against MarketLoop's SQLite data.

    A response succeeds only when it contains every scalar fact required by
    the case's immutable ground truth.  For inventory items, the product name
    and its quantity must occur together, which catches invented products and
    mismatched quantities.  The resulting details are passed to LATS and
    Reflexion as concrete external feedback.
    """

    candidate_contract = """Use only facts supported by the MarketLoop audit data.
State all required numeric values explicitly. For every low-stock product,
write its exact product name and exact current quantity. Do not invent values."""

    def __init__(self, case: TestCase) -> None:
        if case.ground_truth is None:
            raise ValueError(f"Benchmark case {case.id} has no ground-truth function")
        self.case = case
        signature = inspect.signature(case.ground_truth)
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        params = (
            case.params
            if accepts_var_kwargs
            else {
                name: value
                for name, value in case.params.items()
                if name in signature.parameters
            }
        )
        self.truth = case.ground_truth(**params)

    @property
    def source_of_truth(self) -> str:
        return (
            "MarketLoop SQLite database queried by the fixed case ground-truth "
            "function (Orders, Order_Items, Return_Requests, Inventory, and Products)."
        )

    def evaluate(self, state: str) -> EnvironmentFeedback:
        if not isinstance(state, str) or not state.strip():
            return EnvironmentFeedback(success=False, score=0.0, details=["Candidate response is empty."])

        candidate = _normalise(state)
        issues: list[str] = []

        # Inventory records need pair-wise validation; checking a name and a
        # quantity independently would accept a wrong product/quantity pairing.
        inventory_records = self._inventory_records(self.truth)
        for product_name, quantity in inventory_records:
            self._require_inventory_pair(candidate, product_name, quantity, issues)

        for path, expected in _walk_truth(self.truth):
            if self._is_inventory_leaf(path):
                continue
            self._require_scalar(candidate, path, expected, issues)

        total_checks = max(1, len(list(_walk_truth(self.truth))) + len(inventory_records))
        score = max(0.0, round((total_checks - len(issues)) / total_checks, 4))
        return EnvironmentFeedback(success=not issues, score=score, details=issues)

    @staticmethod
    def _inventory_records(truth: Any) -> list[tuple[str, int]]:
        records: list[tuple[str, int]] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if {"product_name", "quantity"}.issubset(value):
                    records.append((str(value["product_name"]), int(value["quantity"])))
                else:
                    for item in value.values():
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(truth)
        return records

    @staticmethod
    def _is_inventory_leaf(path: tuple[str, ...]) -> bool:
        return any(part in {"product_id", "product_name", "quantity"} for part in path)

    @staticmethod
    def _require_inventory_pair(candidate: str, name: str, quantity: int, issues: list[str]) -> None:
        product = _normalise(name)
        match = re.search(re.escape(product) + r".{0,80}", candidate)
        if match is None:
            issues.append(f"Missing low-stock product '{name}' from the database result.")
            return
        nearby = match.group(0)
        if not any(_normalise(variant) in nearby for variant in _number_variants(quantity)):
            issues.append(f"Low-stock product '{name}' must show quantity {quantity}.")

    @staticmethod
    def _require_scalar(candidate: str, path: tuple[str, ...], expected: Any, issues: list[str]) -> None:
        label = path[-1].replace("_", " ") if path else "value"
        if isinstance(expected, bool):
            variants = {str(expected).lower()}
        elif isinstance(expected, (int, float)):
            variants = {_normalise(item) for item in _number_variants(expected)}
        else:
            variants = {_normalise(str(expected))}

        if isinstance(expected, (int, float)):
            # Do not accept 0 merely because a made-up decimal such as
            # ``999.00`` happens to contain that character.
            present = any(
                re.search(r"(?<![0-9.])" + re.escape(variant) + r"(?![0-9.])", candidate)
                for variant in variants
            )
        else:
            present = any(variant in candidate for variant in variants)

        if not present:
            issues.append(f"Missing or incorrect {label}; database value is {expected!r}.")
