"""Manual LATS demo; it must not create an API client during pytest import."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from planning_lab.algorithms.environment import Environment
from planning_lab.algorithms.lats import lats


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    result = lats(
        task="Solve 2 + 2.",
        llm=ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0,
        ),
        environment=Environment(success_threshold=0.6),
        iterations=2,
        n_actions=2,
    )

    print("Success:", result.success)
    print("Output:", result.output)
    print("Best score:", result.best_score)
    print("Iterations:", result.iterations)


if __name__ == "__main__":
    main()
