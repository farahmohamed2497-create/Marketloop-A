from __future__ import annotations

import uuid
from langchain_groq import ChatGroq
from mcp_server import agent_registry
from state_graph.factory import (
    create_graph1,
    create_state,
)


def main() -> None:
    run_id = str(uuid.uuid4())
    agent_registry.ensure_agent("state-graph-agent", display_name="state-graph-agent")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    engine = create_graph1(llm=llm)

    state = create_state(
        run_id=run_id,
        graph_name="decomposition_execution",
        goal="Generate a sales audit report.",
    )

    final_state = engine.run(
        state,
        max_steps=20,
    )

    print("Run ID:", final_state.run_id)
    print("Status:", final_state.status)
    print("Current node:", final_state.current_node)
    print("Transitions:", final_state.transition_count)


if __name__ == "__main__":
    main()