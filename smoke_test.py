from mcp_server.server import MarketLoopMCPServer

server = MarketLoopMCPServer(agent_id="test-agent")
result = server.connect()

print("Connected OK:", result)
print("Tools visible to 'test-agent':", server.list_tools())