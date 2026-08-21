from __future__ import annotations

import sqlite3
import subprocess
import sys

from state_graph.checkpointing.store import CheckpointStore


def test_process_kill_mid_run_then_restart_and_recover(tmp_path):
    db_path = tmp_path / "process_kill.db"

    worker_code = f"""
import sqlite3
import time

from state_graph.core.models import GraphState
from state_graph.checkpointing.store import CheckpointStore

db_path = r"{db_path}"

store = CheckpointStore(
    connection_factory=lambda: sqlite3.connect(db_path)
)

state = GraphState(
    run_id="process-kill-run",
    graph_name="shipping",
    current_node="constrained_react",
    goal="Package is missing",
    transition_count=2,
    data={{
        "subtasks": [
            "check tracking",
            "investigate carrier",
        ]
    }},
)

store.save(state)

print("CHECKPOINT_SAVED", flush=True)

while True:
    time.sleep(1)
"""

    process = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert process.stdout is not None

    line = process.stdout.readline()

    assert "CHECKPOINT_SAVED" in line

    process.kill()
    process.wait()

    store = CheckpointStore(
        connection_factory=lambda: sqlite3.connect(str(db_path))
    )

    recovered = store.load_latest("process-kill-run")

    assert recovered is not None
    assert recovered.current_node == "constrained_react"
    assert recovered.transition_count == 2
    assert recovered.data["subtasks"] == [
        "check tracking",
        "investigate carrier",
    ]