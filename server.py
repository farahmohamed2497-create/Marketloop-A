from db.init_db import initialize_database
from mcp_server.server import MarketLoopMCPServer


if __name__ == "__main__":
    initialize_database()
    MarketLoopMCPServer().run()
