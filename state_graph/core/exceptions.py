from enum import Enum

from pydantic import ValidationError


class StateGraphError(RuntimeError):
    """Base exception for state-graph failures."""


class InvalidTransitionError(StateGraphError):
    """Raised when a graph attempts an invalid transition."""


class CheckpointNotFoundError(StateGraphError):
    """Raised when a requested checkpoint does not exist."""


class ResumeError(StateGraphError):
    """Raised when a graph cannot be resumed safely."""


class HumanInterventionRequired(StateGraphError):
    """Raised when execution must pause for human intervention."""


class FailureKind(str, Enum):
    """Failure categories that require a ticket instead of an automatic retry."""

    TOOL_ERROR = "tool_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"
    UNPLANNED_ERROR = "unplanned_error"


def classify_failure(error: Exception) -> FailureKind:
    """Classify failures that a graph cannot safely retry on its own."""

    if isinstance(error, ValidationError):
        return FailureKind.SCHEMA_VALIDATION_ERROR

    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return FailureKind.TOOL_ERROR

    return FailureKind.UNPLANNED_ERROR
