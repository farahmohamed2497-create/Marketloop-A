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