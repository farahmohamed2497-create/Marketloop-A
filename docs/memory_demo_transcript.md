# Memory Demo Transcript

## Scenario

Customer opens a return request.

Customer:
The item arrived damaged in shipping.

Agent:
Checking the return policy.

Tool:
read_resource(return_policy)

Tool:
process_return_request(order_id=1)

Tool:
generate_sales_audit_report(...)

Several shipment lookup calls occur.

### Rolling Buffer

Old turns are evicted once capacity is exceeded.

### Promote-or-Drop Router

PROMOTE:
Customer states return reason:
"item arrived damaged in shipping"

Reason:
Durable customer fact relevant to future sessions.

FORGET:
Shipment lookup results.

Reason:
Tool output can be re-fetched later.

### Outcome

Important customer context survives through episodic memory while
tool noise is discarded.