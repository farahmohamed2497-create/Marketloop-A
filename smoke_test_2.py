from mcp_server.server import MarketLoopMCPServer
from mcp_server import agent_registry

server = MarketLoopMCPServer(agent_id="test-agent")
server.connect()

print("Before disabling:", server.list_tools())

# simulate what the admin panel does
agent_registry.set_tool_enabled("test-agent", "generate_delay_apology", enabled=False)

print("After disabling (same running server, no restart):", server.list_tools())

# re-enable, to leave things clean
agent_registry.set_tool_enabled("test-agent", "generate_delay_apology", enabled=True)

print("After re-enabling:", server.list_tools())