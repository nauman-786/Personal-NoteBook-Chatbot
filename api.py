from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    thread_has_document,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NAMES_FILE = "conversation_names.json"


def load_names() -> dict:
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_names(names: dict):
    with open(NAMES_FILE, "w") as f:
        json.dump(names, f, indent=2)


@app.get("/api/threads")
def get_threads():
    names = load_names()
    threads = retrieve_all_threads()
    result = []
    for tid in threads:
        result.append({
            "id": str(tid),
            "name": names.get(str(tid), "New chat"),
            "has_document": thread_has_document(str(tid)),
            "document": thread_document_metadata(str(tid)),
        })
    return {"threads": result[::-1]}


@app.post("/api/threads")
def create_thread():
    thread_id = str(uuid.uuid4())
    return {"id": thread_id}


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str):
    names = load_names()
    if thread_id in names:
        del names[thread_id]
        save_names(names)
    return {"ok": True}


@app.patch("/api/threads/{thread_id}/name")
def rename_thread(thread_id: str, body: dict):
    names = load_names()
    names[thread_id] = body.get("name", "New chat")
    save_names(names)
    return {"ok": True}


@app.get("/api/threads/{thread_id}/messages")
def get_messages(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            result.append({"role": "assistant", "content": msg.content})
    return {"messages": result}


@app.post("/api/threads/{thread_id}/upload")
async def upload_pdf(thread_id: str, file: UploadFile = File(...)):
    data = await file.read()
    summary = ingest_pdf(data, thread_id=thread_id, filename=file.filename)
    return summary


@app.post("/api/threads/{thread_id}/chat")
async def chat(thread_id: str, body: dict):
    user_input = body.get("message", "")

    names = load_names()
    if thread_id not in names:
        words = user_input.strip().split()
        name = " ".join(words[:5]).capitalize() or "New chat"
        names[thread_id] = name
        save_names(names)

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    def generate():
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "tool")
                yield f"data: {json.dumps({'type': 'tool', 'name': tool_name})}\n\n"
            elif isinstance(chunk, AIMessage) and chunk.content:
                yield f"data: {json.dumps({'type': 'text', 'content': chunk.content})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=".", html=True), name="static")
