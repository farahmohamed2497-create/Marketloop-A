# MarketLoop State Graphs

This package implements three stateful workflows for the MarketLoop
final project.

## Graph 1 - Return / Refund Resolution

Handles returns that may require:

- return eligibility checks
- policy grounding
- human approval
- waiting for physical return
- inspection
- refund processing

LLM additions:

1. Task Decomposition
2. RAG / Policy Grounding

The graph can pause while waiting for the return and can pause for
human approval when a return exceeds the configured amount threshold.

---

## Graph 2 - Delayed Order Escalation

Handles delayed orders that require:

- order analysis
- candidate action generation
- constrained tool actions
- external carrier response
- escalation
- human approval

LLM additions:

1. LATS / Tree of Thoughts
2. Constrained ReAct

The graph waits for an external carrier response and can return to
the waiting state after an approved retry.

### Why Graph 2 needs a cycle

A delivery investigation cannot safely run as a straight line: after the
agent opens a carrier claim, the carrier may respond hours or days later with
new tracking evidence. Graph 2 therefore transitions from
`constrained_react` to `awaiting_carrier` when it needs that evidence. Once a
carrier response is attached to the persisted state, the graph loops back to
`constrained_react`, which can make a decision from the new evidence without
repeating the original task decomposition. The explicit states and allowed
edges are defined in `state_graph/graph2/design.py`.

---

## Graph 3 - Escalated Return Dispute (Legal / Chargeback Threat)

Handles a Return_Requests case that has already been decided but the
customer is escalating -- threatening a bank chargeback or legal action.
This is a different scope from Graph 1 (Return / Refund Resolution): Graph 1
evaluates a fresh return; Graph 3 only runs once a return is already
closed and the customer refuses to accept the outcome.

LLM additions:

1. Tree of Thoughts (choose which retention argument to lead with)
2. Constrained ReAct (execute only whitelisted dispute-handling tools)

The graph waits for the customer's response to a retention offer (a real
external wait, `awaiting_customer_response`) and can pause separately for
compliance review when the customer raises a legal threat, the confidence
is low, or the offer exceeds the configured limit.

### Graph 3 HITL trigger and resume flow

`dispute_requires_compliance_review` pauses the workflow when any of these
conditions is true: the ReAct confidence is below **0.70**, the proposed
offer is above **500**, the customer makes a legal/chargeback threat, or the
proposal violates policy. The pause writes the complete state to the HITL
request and checkpoint, then enqueues the same snapshot in
`Admin_Task_Queue` for the platform.

The admin records `approve` or `reject` against that request. Calling
`StateGraphEngine.resume(run_id)` reloads the checkpoint and returns to
`dispute_react`, which reads that decision and performs the single final
resolution write; it does not run the LLM again.

### Graph 3: tickets versus HITL

- **HITL** (`dispute_requires_compliance_review`): an *expected* business
  decision -- the customer threatened legal action, or the proposed offer
  needs a human's sign-off. The run is checkpointed with
  `status="waiting"` and `waiting_request_id`; it resumes once an admin
  records `approve`/`reject` through the platform.
- **Failure ticket**: an *unplanned* error -- `sync_dispute_resolution`
  updates `Return_Requests` and then `Audit_Log` in two separate writes.
  If the second write fails after the first already committed
  (`DisputeSyncError`), the run is checkpointed with `status="failed"`
  and a ticket is opened. A blind retry would either re-apply the
  `Return_Requests` update or create a duplicate `Audit_Log` row, so the
  run only resumes once an admin confirms the ticket is resolved
  (`FailureTicketService.resolve_ticket`) and `StateGraphEngine.resume()`
  reloads the checkpoint and finishes only the unfinished node.
---

## Checkpointing

A checkpoint is persisted after each meaningful state transition.

The checkpoint contains:

- run ID
- graph name
- current node
- graph status
- state data
- version
- timestamp

A new process can restore the latest checkpoint using
`StateGraphEngine.recover()`.

---

## Graph 1: tickets versus HITL

Graph 1 (Return / Refund Resolution) uses two deliberately separate pause
paths. They must not be treated as interchangeable.

### Expected decision: HITL

`RefundGraph.lats_node()` calls `HITLNode.pause()` when a refund amount
exceeds the approval threshold, the policy conflicts, or the LLM confidence
is too low. This is an expected business decision, so the run is checkpointed
with `status="waiting"` and `waiting_request_id`. An administrator resolves
the HITL request and supplies the decision before the workflow continues.

    running -> waiting -> admin decision -> resume

### Unplanned error: failure ticket

`StateGraphEngine.step()` catches errors that a Graph 1 node cannot safely
retry. Tool/network failures and schema-validation failures are classified,
the failed state is checkpointed, and `FailureTicketService.create_ticket()`
creates a durable ticket with the node name and saved state. The run uses
`status="failed"` and `waiting_ticket_id`; it cannot resume while the ticket
is still open.

    running -> failed -> ticket(open) -> ticket(resolved) -> resume

After an administrator resolves the failure ticket,
`StateGraphEngine.resume()` reloads the latest checkpoint and executes only
the unfinished node. The crash-recovery test deliberately terminates a
separate process after a checkpoint and verifies that the completed node is
not run again after restart.

---

# Graph 2 — Shipping / Delivery Issue Investigation

## Overview
Graph 2 manages shipping and delivery investigations (damaged, missing, or delayed packages) involving unpredictable third-party carrier responses and contradictory tracking data. It handles both expected human-in-the-loop (HITL) escalations and unexpected runtime failures with robust state checkpointing and recovery.

---

## Core Components & Nodes

- **Task Decomposition (`decompose`)**: Breaks down unstructured customer complaints into concrete subtasks before agent execution. Subtasks are stored in the graph state to persist across checkpoints and recoveries.
- **Constrained ReAct (`constrained_react`)**: Executes the investigation strictly using allowed tools:
  - `check_tracking`
  - `open_carrier_claim`
  - `escalate_to_hitl`
  - *Outcomes*: Can complete successfully, pause for HITL review, or route to failure handling upon errors.

---

## Checkpointing & State Persistence

- **State Persistence**: A complete `GraphState` is saved after every meaningful transition (containing `run_id`, `graph_name`, `current_node`, `status`, `transition_count`, `data`, and `outputs`).
- **Checkpoint Restore**: A new `StateGraphEngine` instance can reopen the database, restore the latest run state, and resume execution without starting from the beginning.
- **No Node Re-execution**: Nodes completed prior to an interruption (e.g., `decompose`) are preserved in the checkpoint and skipped during recovery to avoid duplicated work.

---

## Failure Handling: HITL vs. Failure Tickets

| Feature | Human-in-the-Loop (HITL) | Failure Ticket System |
| **Purpose** | Expected business logic pause | Unexpected technical/execution failure |
| **Triggers** | Decisions requiring human authorization or domain review | Tool crashes, schema validation errors, contradictory API data, unhandled exceptions |
| **Lifecycle** | Awaits operator input $\rightarrow$ resumes flow | `open` $\rightarrow$ `investigating` $\rightarrow$ `resolved` |
| **Resume Point** | Next configured node | Exact failed state restored from checkpoint |

---

## Key Recovery Guarantees

- Automated checkpointing on state transitions.
- Full state restoration across process kills and engine restarts.
- Detection and classification of unexpected runtime errors before ticket generation.
- Failure ticket creation paired with the failed state checkpoint.
- Resumption directly from the persisted failure state upon admin ticket resolution.

---

## Test Suite Coverage

- **`test_checkpoint_write.py`**: Validates checkpoint persistence, node state, and transition counters after valid steps.
- **`test_checkpoint_restore.py`**: Tests database reopening and state restoration across new engine instances.
- **`test_failure_detection.py`**: Verifies classification of schema and tool errors as execution failures.
- **`test_failure_ticket.py`**: Ensures tickets capture metadata (`run_id`, `node`, `error`, `status`).
- **`test_resume_after_ticket_resolution.py`**: Validates the end-to-end failure $\rightarrow$ ticket $\rightarrow$ resolution $\rightarrow$ resume path.
- **`test_process_kill_resume.py`**: Confirms recovery after hard process termination.
- **`test_no_reexecution_after_recovery.py`**: Confirms previously completed nodes do not re-run after restart.

##  Platform: RAG Document Management + HITL/Ticket Resolution

```
 Platform
├── Backend / Integration
│   ├── RAG document lifecycle   (admin_api/main.py, RAG/document_store.py)
│   ├── HITL endpoints           (admin_api/main.py, state_graph/hitl/node.py)
│   └── Ticket endpoints         (admin_api/main.py, state_graph/tickets/service.py)
│
├── Testing
│   ├── Platform API tests       (tests/platform/, 11 passing)
│   ├── RAG add/remove E2E       (test_rag_documents.py)
│   ├── HITL E2E                 (test_hitl_and_tickets.py)
│   └── Ticket E2E               (test_hitl_and_tickets.py)
│
└── UI
    ├── RAG management           (admin_ui/index.html — "RAG Documents" tab)
    ├── HITL inbox                (admin_ui/index.html — "HITL Inbox" tab)
    └── Ticket board              (admin_ui/index.html — "Tickets" tab)
```

### Backend / Integration

- `GET/POST /api/rag/documents`, `DELETE /api/rag/documents/{id}` — chunk,
  embed, and index a document into the shared VectorStore + BM25
  KeywordStore; removal purges both stores and re-checks out on delete.
  Every mutation bumps `Rag_Store_Meta.version` in the shared DB.
- `RAG/rag_pipeline.py` checks that version before every `answer_with_*`
  call and drops its in-memory cache if it's stale — this is what makes
  an admin's add/remove reach a *running* agent process on its next
  query, not just after a restart, since the version lives in the DB
  both processes already read.
- `GET /api/hitl/tasks`, `GET /api/hitl/tasks/{id}`,
  `POST /api/hitl/tasks/{id}/resolve` — lists pending human-review
  requests across all three graphs and records the admin's decision.
  Resolving calls `state_graph.engine_registry.resume_run(run_id,
  graph_name)`, which resumes the actual paused run via
  `StateGraphEngine.resume()`. There is no auto-approve path anywhere —
  a run only advances because this endpoint was hit.
- `GET /api/tickets`, `GET /api/tickets/{id}`, `PATCH /api/tickets/{id}`
  — same resume mechanism for unplanned failures. Moving a ticket to
  `resolved` resumes the run from its last checkpoint (the exact node
  it failed on); moving it to `investigating` does not resume anything.
- `state_graph/engine_registry.py` (new) — maps `graph_name` ("refund" /
  "shipping" / "dispute") to a live `StateGraphEngine`, built lazily.

**Bugs fixed while wiring this up** (pre-existing, unrelated to F2 scope
but blocking it):
- `state_graph/factory.py`: `create_graph1()` referenced
  `build_graph1/2/3` without importing them and passed `environment=`
  instead of `llm=` — could not have run.
- `state_graph/checkpointing/recovery.py`: dead import of a module that
  doesn't exist (`state_graph.failure_tickets.service`).
- `state_graph/hitl/node.py`: `HITL_Requests` was only created lazily
  inside `create_request()`, so a lookup on an ID that was never
  created threw `sqlite3.OperationalError`. Now created in `__init__`.
- `RAG/vector_store.py`: `remove_by_metadata()` didn't purge
  `_id_map`/`_reverse_id_map`, so `.size` kept counting deleted vectors
  and search crashed after deleting the last live document. Fixed.

### Testing

`tests/platform/` — 11 tests, all passing against the real endpoints,
real HITLNode/FailureTicketService/RagDocumentStore (isolated per test
on a throwaway sqlite DB), with only the embedding model and the LLM
resume call faked:

- add a document → verify the agent retrieves it on its next query
- remove a document → verify the agent stops retrieving it
- removing an unknown document → 404
- a HITL task appears on the platform when a graph pauses
- an admin resolving a HITL task records the decision and resumes the
  run
- resolving an unknown HITL task → 404; an invalid decision → 400 and
  no resume
- a ticket appears when a node fails
- resolving a ticket resumes the run from its checkpoint
- moving a ticket to "investigating" does **not** resume it
- resolving an unknown ticket → 404

Still open: the demo clip (recording, not code) showing this working
live through the UI end to end.

### UI

`admin_ui/index.html` — three new tabs alongside the existing
Agents & Tools tab:

- **RAG Documents** — list with chunk count + added date, an
  "Add document" form (filename + text), and a remove action per row.
- **HITL Inbox** — pending requests with the paused graph's persisted
  state shown inline, and Approve/Reject buttons that call the real
  resolve endpoint.
- **Tickets** — open/investigating/resolved board; each card has a
  "View checkpoint state" toggle showing the failed run's persisted
  state, a "Start investigating" action, and a "Mark resolved & resume"
  action that prompts for a resolution note and resumes the run.

No mock data anywhere — every view reads from and writes to the live
admin API.



## Running the package

From the repository root:

```bash
python -m state_graph
