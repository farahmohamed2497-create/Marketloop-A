from planning_lab.mcp_executor import MarketLoopMCPExecutor

executor = MarketLoopMCPExecutor(allow_mutations=True)
print(executor.execute("generate_sales_audit_report", {"start_date": "2026-01-01", "end_date": "2026-01-31"}))