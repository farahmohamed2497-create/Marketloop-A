import asyncio

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from pydantic import BaseModel, ConfigDict
from typing import Callable, Optional


class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str


MCP_SERVERS_CONFIG = {
    "main_server": {
        "transport": "stdio",
        "command": "py",
        "args": ["-m", "mcp_server.server"],
    },
}



def make_mcp_executor(
    servers_config: dict = MCP_SERVERS_CONFIG,
) -> Callable[[BaseChatModel, str, str, str], str]:
    """
    Real executor: fetches the tools from the MCP server (including the
    database, if it's registered as an MCP server) once, and builds a
    react-style agent that can pick and execute the right tool for each
    sub-task coming from dynamic_decomposition.
 
    Returns an executor with the same signature as before,
    Callable[[llm, goal, task, observation], str], so it drops straight into
    dynamic_decomposition without any other changes to this file.
    """
    tools_cache: dict[str, list] = {}

    async def _get_tools():
        if "tools" not in tools_cache:
            client = MultiServerMCPClient(servers_config)
            tools = await client.get_tools()
            if not tools:
                raise RuntimeError(
                    "No MCP tools were found. Please check MCP_SERVERS_CONFIG and make sure the server is running."
                )
            tools_cache["tools"] = tools
        return tools_cache["tools"]

    async def _run(llm: BaseChatModel, goal: str, task: str, observation: str) -> str:
        tools = await _get_tools()
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "Execute the next adaptive sub-task using the available tools. "
                "Use the tool results as ground truth; do not hallucinate data."
            ),
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}",
                    },
                ]
            }
        )
        content = result["messages"][-1].content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The MCP-backed agent returned an empty or unsupported response")
        return content.strip()

    def executor(llm: BaseChatModel, goal: str, task: str, observation: str) -> str:
        return asyncio.run(_run(llm, goal, task, observation))

    return executor


def _default_executor(llm: BaseChatModel, goal: str, task: str, observation: str) -> str:
    """
    Fallback executor: asks the LLM to describe what it would do, without
    calling any real tool. Kept for quick tests when no MCP server is
    available. Prefer `make_mcp_executor()` for real runs.
    """
    response = llm.invoke([
        ("system", "Execute the next adaptive sub-task using the observations provided."),
        ("human", f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}"),
    ], temperature=0.2)
    result = response.content
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    return result.strip()


def dynamic_decomposition(
    goal: str,
    llm: BaseChatModel,
    max_steps: int = 4,
    executor: Optional[Callable[[BaseChatModel, str, str, str], str]] = None,
) -> list[tuple[str, str]]:
    """
    `executor` is the function that actually carries out each sub-task once
    decided. Defaults to an LLM-only stand-in; pass `make_mcp_executor()`
    (or your own) to actually call MCP tools / the database.
    """
    executor = executor or _default_executor
    history: list[tuple[str, str]] = []

    for step in range(max_steps):
        observation = "\n".join(f"{task}: {result}" for task, result in history) or "None"
        decision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke([
            ("system", "You are an adaptive planner. Use prior observations before deciding what comes next."),
            ("human", f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task. Set done to true only when the goal is met.
When done is true, use an empty string for next_task."""),
        ], temperature=0.1)

        if decision.done:
            break

        task = decision.next_task.strip()
        if not task:
            raise ValueError(f"Dynamic planner omitted next_task at step {step + 1}")

        result = executor(llm, goal, task, observation)
        history.append((task, result))

    return history




def get_llm() -> BaseChatModel:
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)



if __name__ == "__main__":
    llm = get_llm()
    executor = make_mcp_executor()  
    goal = (
    "Analyze the sales performance for January 2026, check the return "
    "rate and low-stock products, identify operational risks, and "
    "produce a final management summary."
)
    history = dynamic_decomposition(goal, llm, max_steps=4, executor=executor)
    for task, result in history:
        print(f"\n--- Task: {task} ---\n{result}")