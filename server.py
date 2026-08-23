from db.init_db import initialize_database
from mcp_server.server import MarketLoopMCPServer
from mcp_server.tools import agent_registry
import os

if __name__ == "__main__":
    initialize_database()
    MarketLoopMCPServer(
        agent_id=os.getenv("MARKETLOOP_AGENT_ID", agent_registry.DEFAULT_AGENT_ID),
    ).run()
