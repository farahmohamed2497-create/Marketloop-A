from __future__ import annotations

import subprocess
import sys

from state_graph.checkpointing.store import CheckpointStore
import sqlite3


def test_process_kill_mid_run_then_restart_and_recover(tmp_path):
    db_path = tmp_path / "process_kill.db"

    worker_code = f"""
import sqlite3
import time

from state_graph.core.models import GraphState
from state_graph.checkpointing.store import CheckpointStore

db_path = r"{db_path}"

store = CheckpointStore(connection_factory=lambda: sqlite3.connect(db_path))

state = GraphState(
    run_id="process-kill-run",
    graph_name="dispute",
    current_node="dispute_react",
    goal="Customer threatens a chargeback on return #42",
    transition_count=2,
    data={{"retention_strategy": {{"strategy": "offer partial refund"}}}},
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

    store = CheckpointStore(connection_factory=lambda: sqlite3.connect(str(db_path)))
    recovered = store.load_latest("process-kill-run")

    assert recovered is not None
    assert recovered.current_node == "dispute_react"
    assert recovered.transition_count == 2
    assert recovered.data["retention_strategy"]["strategy"] == "offer partial refund"