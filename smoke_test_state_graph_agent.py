from mcp_server import agent_registry

agent_registry.ensure_agent("state-graph-agent", display_name="state-graph-agent")
print("registered OK")