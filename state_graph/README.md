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

---

## Graph 3 - Inventory Discrepancy Recovery

Handles inventory discrepancies between system records and warehouse
counts.

LLM additions:

1. RAG / Inventory Policy Grounding
2. Constrained ReAct

The workflow requires warehouse confirmation and human approval before
changing the system-of-record inventory quantity.

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

## Running the package

From the repository root:

```bash
python -m state_graph
