from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from ..models import Thought


class ThoughtCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Small models sometimes propose more than requested. Keep the schema
    # permissive, then retain the best two candidates in the search loop.
    candidates: list[str] = Field(min_length=1, max_length=10)


class ThoughtEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


def _structured_or_default(runnable, messages, *, temperature: float, default):
    """Keep one malformed provider function call from aborting a search."""
    try:
        return runnable.invoke(messages, temperature=temperature)
    except Exception:
        return default


def _expand_and_score(
    problem: str,
    parent: Thought,
    llm: BaseChatModel,
    prune_threshold: float,
) -> list[Thought]:
    """Use the toolkit's generate/evaluate loop for one search-tree node."""
    generated = _structured_or_default(
        llm.with_structured_output(
            ThoughtCandidates,
            method=getattr(llm, "structured_output_method", "json_schema"),
        ), [
        ("system", "Generate distinct candidate next steps for Tree-of-Thoughts search."),
        ("human", f"""Problem: {problem}
Partial path: {parent.state}
Propose two distinct promising continuations."""),
        ],
        temperature=0.5,
        default=ThoughtCandidates(candidates=[parent.state]),
    )

    children: list[Thought] = []
    for state in generated.candidates[:2]:
        judged = _structured_or_default(
            llm.with_structured_output(
                ThoughtEvaluation,
                method=getattr(llm, "structured_output_method", "json_schema"),
            ), [
           (
    "system",
    """Independently evaluate a partial solution.

Return ONLY valid JSON in this format:
{
  "score": 0.0,
  "rationale": "brief explanation"
}

The score must be between 0 and 1.""",
),
            ("human", f"""Problem: {problem}
Candidate path: {state}
Score correctness, feasibility, and progress. Do not reward confident wording."""),
            ],
            temperature=0.1,
            default=ThoughtEvaluation(score=0.0, rationale="Malformed provider response."),
        )
        if judged.score >= prune_threshold:
            children.append(
                Thought(state=state, score=judged.score, rationale=judged.rationale)
            )
    return sorted(children, key=lambda item: item.score, reverse=True)


def tree_of_thoughts(
    problem: str,
    llm: BaseChatModel,
    depth: int = 2,
    beam_width: int = 2,
    search_strategy: str = "bfs",
    prune_threshold: float = 0.0,
) -> list[Thought]:
    """Search candidate reasoning paths with BFS or DFS and explicit pruning.

    This extends the reference toolkit's candidate-generation and
    self-evaluation calls.  BFS keeps a global beam at each level; DFS follows
    a promising branch before exploring its siblings.  In either strategy,
    candidates below ``prune_threshold`` are removed before expansion.
    """
    if depth < 1 or beam_width < 1:
        raise ValueError("depth and beam_width must be positive")
    if search_strategy not in {"bfs", "dfs"}:
        raise ValueError("search_strategy must be 'bfs' or 'dfs'")
    if not 0.0 <= prune_threshold <= 1.0:
        raise ValueError("prune_threshold must be between zero and one")

    frontier = [Thought(state="Start", score=0.5, rationale="root")]
    if search_strategy == "bfs":
        for _ in range(depth):
            candidates = [
                child
                for parent in frontier
                for child in _expand_and_score(problem, parent, llm, prune_threshold)
            ]
            frontier = sorted(candidates, key=lambda item: item.score, reverse=True)[:beam_width]
            if not frontier:
                break
        return frontier

    stack: list[tuple[Thought, int]] = [(frontier[0], 0)]
    completed: list[Thought] = []
    while stack:
        parent, level = stack.pop()
        if level == depth:
            completed.append(parent)
            continue
        children = _expand_and_score(problem, parent, llm, prune_threshold)[:beam_width]
        stack.extend((child, level + 1) for child in reversed(children))
    return sorted(completed, key=lambda item: item.score, reverse=True)[:beam_width]
