-- =========================================================
-- Persistent State Graph
-- =========================================================

CREATE TABLE IF NOT EXISTS State_Checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL,

    graph_name TEXT NOT NULL,

    current_node TEXT NOT NULL,

    status TEXT NOT NULL,

    goal TEXT,

    data TEXT NOT NULL DEFAULT '{}',

    outputs TEXT NOT NULL DEFAULT '{}',

    transition_count INTEGER NOT NULL DEFAULT 0,

    last_error TEXT,

    waiting_request_id TEXT,

    waiting_ticket_id TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX IF NOT EXISTS idx_state_checkpoint_run
ON State_Checkpoints(run_id);


CREATE INDEX IF NOT EXISTS idx_state_checkpoint_run_updated
ON State_Checkpoints(run_id, updated_at);


-- =========================================================
-- Failure Tickets
-- =========================================================

CREATE TABLE IF NOT EXISTS Failure_Tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL,

    graph_name TEXT NOT NULL,

    node_name TEXT NOT NULL,

    error TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'open',

    resolution TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    resolved_at TIMESTAMP
);


CREATE INDEX IF NOT EXISTS idx_failure_ticket_run
ON Failure_Tickets(run_id);


CREATE INDEX IF NOT EXISTS idx_failure_ticket_status
ON Failure_Tickets(status);


-- =========================================================
-- Human-in-the-loop Requests
-- =========================================================

CREATE TABLE IF NOT EXISTS HITL_Requests (
    request_id TEXT PRIMARY KEY,

    run_id TEXT NOT NULL,

    graph_name TEXT NOT NULL,

    reason TEXT NOT NULL,

    state TEXT NOT NULL,

    decision TEXT,

    status TEXT NOT NULL DEFAULT 'pending',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    resolved_at TIMESTAMP
);