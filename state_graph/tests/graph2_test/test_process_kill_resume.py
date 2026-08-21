from __future__ import annotations

import json
import subprocess
import sys


def test_process_kill_recovery(tmp_path):
    db_path = tmp_path / "process_kill.db"

    worker = r"""
from __future__ import annotations

import sys

from state_graph.checkpointing.store import CheckpointStore
from state_graph.core.models import GraphState


db_path = sys.argv[1]

def connection_factory():
    import sqlite3
    return sqlite3.connect(db_path)

store = CheckpointStore(
    connection_factory=connection_factory
)

state = GraphState(
    run_id="process-kill-run",
    graph_name="shipping",
    current_node="awaiting_carrier",
    goal="Package is missing",
    transition_count=1,
    data={
        "completed_steps": [
            "check_tracking"
        ]
    },
)

store.save(state)

print("CHECKPOINT_CREATED", flush=True)

# Simulate process termination after checkpoint creation.
raise SystemExit(137)
"""

    first_process = subprocess.run(
        [
            sys.executable,
            "-c",
            worker,
            str(db_path),
        ],
        capture_output=True,
        text=True,
    )

    assert "CHECKPOINT_CREATED" in first_process.stdout

    restore_script = r"""
from __future__ import annotations

import json
import sys

from state_graph.checkpointing.store import CheckpointStore


db_path = sys.argv[1]

def connection_factory():
    import sqlite3
    return sqlite3.connect(db_path)

store = CheckpointStore(
    connection_factory=connection_factory
)

state = store.load_latest(
    "process-kill-run"
)

assert state is not None
assert state.current_node == "awaiting_carrier"

# The completed transition must still be present.
assert "check_tracking" in state.data["completed_steps"]

print(
    json.dumps(
        {
            "run_id": state.run_id,
            "node": state.current_node,
            "completed_steps": state.data["completed_steps"],
        }
    )
)
"""

    second_process = subprocess.run(
        [
            sys.executable,
            "-c",
            restore_script,
            str(db_path),
        ],
        capture_output=True,
        text=True,
    )

    assert second_process.returncode == 0

    restored = json.loads(
        second_process.stdout
    )

    assert restored["run_id"] == "process-kill-run"
    assert restored["node"] == "awaiting_carrier"
    assert "check_tracking" in restored["completed_steps"]