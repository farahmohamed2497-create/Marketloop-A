"""Compatibility wrapper for return-processing tools with elicitation support."""

from __future__ import annotations

from ..db import get_connection
from .orders import ElicitationContext, process_return_request

__all__ = ["ElicitationContext", "process_return_request", "get_connection"]
