# F3 demo runbook

Run the platform from the repository root:

```powershell
.\.venv\Scripts\python.exe -m platform_app.app
```

Open `http://127.0.0.1:8080` and select an agent. Send a question; the
conversation is stored under the generated agent-specific session ID.
Switching away and back restores that agent's history. The RAG, State Graph,
and Planning selections call their existing project services through F3's
routing layer.

For live RAG, State Graph, and Planning responses, set `GROQ_API_KEY` in
`.env`. Without it the backend returns the provider configuration error
instead of generating an answer.
