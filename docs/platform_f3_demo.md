# F3 demo runbook

Run the platform from the repository root:

```powershell
.\.venv\Scripts\python.exe -m platform_app.app
```

Open `http://127.0.0.1:8080` and select **Memory & RAG**. Send a question;
the conversation is stored under the generated agent-specific session ID.
Switching away and back restores that history. The **State Graph** and
**Planning** buttons intentionally remain disabled until their owners expose
chat-runner adapters; F3 does not reimplement those graph/planning concerns.

For a live RAG response, set `GROQ_API_KEY` in `.env`. Without it the backend
returns the provider configuration error instead of generating an answer.
