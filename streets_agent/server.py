from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from streets_agent.browser import (
    BrowserSessionInfo,
    assert_browserbase_config,
    create_browserbase_session,
    release_session,
    required_env,
)
from streets_agent.graph import standalone_graph

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "public"
TARGET_URL = os.getenv("TARGET_URL", "https://www.google.com")
RUN_TIMEOUT_SECONDS = int(os.getenv("RUN_TIMEOUT_SECONDS", "300"))


class RunRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=2_000)
    start_url: str | None = Field(default=None, max_length=2_000)


@dataclass
class RunRecord:
    run_id: str
    session: BrowserSessionInfo
    events: list[dict[str, Any]] = field(default_factory=list)
    clients: set[WebSocket] = field(default_factory=set)
    status: str = "starting"
    resume_future: asyncio.Future[dict[str, Any]] | None = None
    task: asyncio.Task[None] | None = None


runs: dict[str, RunRecord] = {}


async def emit(record: RunRecord, payload: dict[str, Any]) -> None:
    record.events.append(payload)
    event_type = payload.get("type")
    if event_type == "hitl-request":
        record.status = "paused"
    elif event_type in {"submitted", "cancelled", "error"}:
        record.status = "finished"
    else:
        record.status = "running"
    stale: list[WebSocket] = []
    for client in record.clients:
        try:
            await client.send_json(payload)
        except Exception:
            stale.append(client)
    for client in stale:
        record.clients.discard(client)


def interrupt_value(chunk: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = chunk.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", None)
    if value is None and isinstance(first, dict):
        value = first.get("value")
    return value if isinstance(value, dict) else None


async def stream_graph_once(record: RunRecord, graph_input: dict[str, Any] | Command) -> dict[str, Any] | None:
    waiting: dict[str, Any] | None = None
    config = {"configurable": {"thread_id": record.run_id}, "recursion_limit": 150}
    async for chunk in standalone_graph.astream(graph_input, config=config, stream_mode="updates"):
        if payload := interrupt_value(chunk):
            waiting = payload
            continue
        for update in chunk.values():
            if not isinstance(update, dict):
                continue
            for activity in update.get("activity", []):
                await emit(record, activity)
    return waiting


async def run_graph(record: RunRecord, instruction: str, start_url: str | None = None) -> None:
    graph_input: dict[str, Any] | Command = {
        "instruction": instruction,
        "start_url": start_url or TARGET_URL,
        "session_id": record.session.session_id,
        "history": [],
        "activity": [],
        "fields": [],
        "steps": 0,
        "contains_human_data": False,
    }
    try:
        while True:
            waiting = await stream_graph_once(record, graph_input)
            if not waiting:
                break
            loop = asyncio.get_running_loop()
            record.resume_future = loop.create_future()
            await emit(record, {"type": "hitl-request", "ts": _utc_now(), **waiting})
            try:
                response = await asyncio.wait_for(record.resume_future, timeout=RUN_TIMEOUT_SECONDS)
            except TimeoutError:
                await emit(
                    record,
                    {"type": "error", "message": "Run timed out while waiting for human input.", "ts": _utc_now()},
                )
                break
            finally:
                record.resume_future = None
            graph_input = Command(resume=response)
    except Exception as exc:
        await emit(record, {"type": "error", "message": str(exc), "ts": _utc_now()})
    finally:
        await release_session(record.session.session_id)
        asyncio.create_task(remove_run_later(record.run_id))


async def remove_run_later(run_id: str) -> None:
    await asyncio.sleep(600)
    runs.pop(run_id, None)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    for record in list(runs.values()):
        if record.resume_future and not record.resume_future.done():
            record.resume_future.set_result({"decision": "reject"})
        if record.task and not record.task.done():
            record.task.cancel()
    await asyncio.gather(
        *(release_session(record.session.session_id) for record in runs.values()), return_exceptions=True
    )


app = FastAPI(title="Streets LangGraph Browser Agent", lifespan=lifespan)


@app.get("/config")
async def config() -> dict[str, str]:
    return {"targetUrl": TARGET_URL, "demoFormUrl": "demo://pizza-form"}


@app.post("/run", status_code=201)
async def create_run(request: RunRequest) -> dict[str, str]:
    instruction = request.instruction.strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="An instruction is required.")
    try:
        assert_browserbase_config()
        required_env("OPENAI_API_KEY")
        session = await asyncio.to_thread(create_browserbase_session)
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None) or 500
        raise HTTPException(status_code=status if 400 <= status < 600 else 500, detail=str(exc)) from exc
    run_id = str(uuid4())
    record = RunRecord(run_id=run_id, session=session)
    record.events.append(
        {
            "type": "session",
            "liveViewUrl": session.live_view_url,
            "sessionId": session.session_id,
        }
    )
    runs[run_id] = record
    record.task = asyncio.create_task(run_graph(record, instruction, request.start_url))
    return {"runId": run_id, "liveViewUrl": session.live_view_url}


@app.websocket("/ws")
async def run_socket(websocket: WebSocket, runId: str) -> None:  # noqa: N803 - protocol field is camelCase
    record = runs.get(runId)
    if not record:
        await websocket.close(code=4404, reason="Run not found")
        return
    await websocket.accept()
    record.clients.add(websocket)
    try:
        for item in record.events:
            await websocket.send_json(item)
        while True:
            message = await websocket.receive_json()
            if message.get("runId") != record.run_id or record.status != "paused":
                continue
            future = record.resume_future
            if not future or future.done():
                continue
            if message.get("type") == "confirm":
                value = message.get("value")
                future.set_result({"decision": "approve", "value": str(value)[:2_000] if value is not None else None})
            elif message.get("type") == "cancel":
                future.set_result({"decision": "reject"})
    except WebSocketDisconnect:
        pass
    finally:
        record.clients.discard(websocket)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")
