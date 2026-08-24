from __future__ import annotations

from pathlib import Path

from aiohttp import web

from .agents import AgentRouter
from .sessions import ChatSessionStore

STATIC_DIR = Path(__file__).with_name("static")
ROUTER_KEY: web.AppKey[AgentRouter] = web.AppKey("router", AgentRouter)
STORE_KEY: web.AppKey[ChatSessionStore] = web.AppKey("store", ChatSessionStore)


def create_app(*, router: AgentRouter | None = None, store: ChatSessionStore | None = None) -> web.Application:
    app = web.Application()
    app[ROUTER_KEY] = router or AgentRouter()
    app[STORE_KEY] = store or ChatSessionStore()
    app.router.add_get("/", index)
    app.router.add_static("/static", STATIC_DIR)
    app.router.add_get("/api/agents", list_agents)
    app.router.add_post("/api/chat/messages", post_message)
    app.router.add_get("/api/chat/sessions/{session_id}", get_history)
    return app


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def list_agents(request: web.Request) -> web.Response:
    return web.json_response({"agents": request.app[ROUTER_KEY].list_agents()})


async def post_message(request: web.Request) -> web.Response:
    body = await request.json()
    agent_id = str(body.get("agent_id", "")).strip()
    message = str(body.get("message", "")).strip()
    if not agent_id or not message:
        return web.json_response({"error": "agent_id and message are required."}, status=400)
    store = request.app[STORE_KEY]
    try:
        session_id = store.get_or_create(agent_id=agent_id, session_id=body.get("session_id"))
        store.add_message(session_id=session_id, role="user", content=message)
        reply = request.app[ROUTER_KEY].route(agent_id, message)
        store.add_message(session_id=session_id, role="assistant", content=reply)
    except (ValueError, RuntimeError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"session_id": session_id, "agent_id": agent_id, "reply": reply,
                              "history": store.history(session_id)})


async def get_history(request: web.Request) -> web.Response:
    return web.json_response({"history": request.app[STORE_KEY].history(request.match_info["session_id"])})


def main() -> None:
    web.run_app(create_app(), host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
