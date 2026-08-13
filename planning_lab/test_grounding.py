from langchain_groq import ChatGroq

from .routing import _run_sales_grounding


TASK = """
Create a sales audit for January 2026
from 2026-01-01 to 2026-01-31.
"""

# Deliberately wrong factual claim.
# The database-backed validator should catch this.
DRAFT = """
January 2026 Sales Audit

Total orders: 0
Total revenue: $0.00
Units sold: 0
Average order value: $0.00
Return rate: 0%

Low-stock items:
- Air Fryer: 15 units
- Dell Laptop: 20 units
- iPhone: 8 units

The January 2026 sales audit shows no recorded sales activity during the reporting period.
"""
def run_ungrounded_critique(llm, task, draft):
    response = llm.invoke(
        [
            (
                "system",
                """You are an ungrounded critic.

You have access ONLY to the task and the draft.

You may flag:
- explicit contradictions inside the draft;
- missing sections explicitly required by the task;
- formatting or instruction-following problems.

You MUST NOT infer facts from the real world.
You MUST NOT assume that a claim is true or false based on
outside knowledge.
You MUST NOT use databases, tools, or external evidence.

If there is no explicit contradiction or structural problem,
return exactly:
PASS

Otherwise return:
FAIL: <reason>""",
            ),
            (
                "human",
                f"""Task:
{task}

Draft:
{draft}""",
            ),
        ],
        temperature=0,
    )

    return response.content.strip()

def run_grounded_critique(llm: ChatGroq, task: str, draft: str):
    """
    Grounded critique:
    obtains evidence from the real SQLite database first.
    """

    grounded = _run_sales_grounding(
        task=task,
        draft=draft,
    )

    evidence_text = (
        "\n".join(f"- {item}" for item in grounded.evidence)
        if grounded.evidence
        else "- No inconsistencies found."
    )

    response = llm.invoke(
        [
            (
                "system",
                """You are a grounded critic.

The supplied grounded evidence comes from the real database
and is the source of truth for factual sales claims.

Do not ignore database evidence.

Return:
PASS
or
FAIL: <reason>""",
            ),
            (
                "human",
                f"""Task:
{task}

Draft:
{draft}

SOURCE OF TRUTH:
{grounded.source_of_truth}

GROUNDED DATABASE EVIDENCE:
{evidence_text}""",
            ),
        ],
        temperature=0,
    )

    return grounded, response.content.strip()


def main():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    print("=" * 70)
    print("UNGROUNDED CRITIQUE")
    print("=" * 70)

    ungrounded_result = run_ungrounded_critique(
        llm,
        TASK,
        DRAFT,
    )

    print(ungrounded_result)

    print("\n" + "=" * 70)
    print("GROUNDED CRITIQUE")
    print("=" * 70)

    grounded_result, grounded_critique = run_grounded_critique(
        llm,
        TASK,
        DRAFT,
    )

    print("Source of truth:")
    print(grounded_result.source_of_truth)

    print("\nEvidence:")
    for item in grounded_result.evidence:
        print(f"- {item}")

    print("\nCritique:")
    print(grounded_critique)

    print("\n" + "=" * 70)
    print("EXPECTED RESULT")
    print("=" * 70)

    print(
        """
The grounded version should detect the incorrect revenue claim,
because it compares the draft against the real SQLite database.

The ungrounded version has no database evidence, so it may miss
the factual error and focus only on writing quality/completeness.
"""
    )


if __name__ == "__main__":
    main()