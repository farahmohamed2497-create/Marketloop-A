-- =========================================================
-- Agents (F1 — Admin: Tool & Agent Management)
--
-- Represents an "agent" as a first-class entity the admin
-- panel can see and manage, independent of the role-based
-- filtering already in mcp_server/server.py.
-- =========================================================

CREATE TABLE IF NOT EXISTS Agents (
    agent_id TEXT PRIMARY KEY,

    display_name TEXT NOT NULL,

    description TEXT,

    -- e.g. "memory_rag", "planning", "state_graph:claims", etc.
    agent_type TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    last_seen_at TIMESTAMP
);


-- =========================================================
-- Agent_Tools
--
-- Per-agent tool enablement. Absence of a row for
-- (agent_id, tool_name) means "enabled" (opt-out model) —
-- so newly discovered tools are visible by default and an
-- admin explicitly disables the ones an agent shouldn't have.
-- =========================================================

CREATE TABLE IF NOT EXISTS Agent_Tools (
    agent_id TEXT NOT NULL,

    tool_name TEXT NOT NULL,

    enabled INTEGER NOT NULL DEFAULT 1,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_by TEXT,

    PRIMARY KEY (agent_id, tool_name),

    FOREIGN KEY (agent_id) REFERENCES Agents(agent_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_tools_agent
ON Agent_Tools(agent_id);