from __future__ import annotations


def requires_human_intervention(
    score: float,
    *,
    threshold: float = 0.70,
) -> bool:
    """
    Return True when confidence is below the HITL threshold.
    """

    return score < threshold