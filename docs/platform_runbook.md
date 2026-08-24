# Platform runbook and demonstration transcript

Install the complete dependency set once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the two product surfaces in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn admin_api.main:app --port 8001
.\.venv\Scripts\python.exe -m platform_app.app
```

Open `http://127.0.0.1:8080` for user chat and use **Admin console** to open
the admin API. The static admin interface is `admin_ui/index.html`; serve it
with any static web server while the API is running.

## Required live-demo sequence

1. In the admin UI, disable a tool for an agent; confirm its live MCP call is
   rejected. Re-enable it and confirm the call works.
2. Add a RAG document, ask the RAG agent about its unique text, remove it, and
   ask again to show that retrieval changed.
3. Submit a graph input whose confidence/amount/policy triggers HITL. Open the
   pending request in the admin API/UI, inspect saved state, approve/reject,
   and show the resumed run ID/status.
4. Trigger the documented tool/schema failure. Open its ticket, mark it
   resolved, and show that resume starts at the saved failed node.
5. Start a graph, terminate its process after a checkpoint, restart, and run
   the crash-recovery test to show the completed node was not repeated.

Record these five steps as the submission demo. Do not record or display
`.env` values or API keys.
