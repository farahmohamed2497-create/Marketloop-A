from planning_lab.routing import PlanningMethod, route_sales_audit_subtask


def test_sales_audit_router_selects_methods_by_subtask_shape():
    assert route_sales_audit_subtask("Retrieve audit data from the MCP report") is PlanningMethod.MCP
    assert route_sales_audit_subtask("Compare alternative restock recommendations") is PlanningMethod.TREE_OF_THOUGHTS
    assert route_sales_audit_subtask("Retry validation after an approval failure") is PlanningMethod.REFLEXION
    assert route_sales_audit_subtask("Commit an inventory adjustment") is PlanningMethod.LATS
    assert route_sales_audit_subtask("Summarize the completed audit") is PlanningMethod.PLAN_AND_SOLVE