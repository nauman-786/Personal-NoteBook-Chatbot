from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
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
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_names(names: dict):
    with open(NAMES_FILE, "w") as f:
        json.dump(names, f, indent=2)


def is_authorized(thread_id: str, x_user_id: str, names: dict) -> bool:
    """Helper to check if the current user owns the thread."""
    info = names.get(thread_id)
    if isinstance(info, dict):
        return info.get("user_id") == x_user_id
    # If it's an old string-based chat from before the update, allow access for now
    return True


@app.get("/api/threads")
def get_threads(x_user_id: Optional[str] = Header(None)):
    names = load_names()
    threads = retrieve_all_threads()
    result = []
    for tid in threads:
        tid_str = str(tid)
        info = names.get(tid_str)
        
        # Handle new dictionary format vs old string format
        if isinstance(info, dict):
            thread_user_id = info.get("user_id")
            thread_name = info.get("name", "New chat")
        else:
            thread_user_id = None
            thread_name = info if isinstance(info, str) else "New chat"
            
        # Multi-Tenant Filter: Only return threads belonging to THIS user
        if x_user_id and thread_user_id == x_user_id:
            result.append({
                "id": tid_str,
                "name": thread_name,
                "has_document": thread_has_document(tid_str),
                "document": thread_document_metadata(tid_str),
            })
            
    return {"threads": result[::-1]}


@app.post("/api/threads")
def create_thread(x_user_id: Optional[str] = Header(None)):
    thread_id = str(uuid.uuid4())
    names = load_names()
    # Save the thread name AND the user_id that owns it
    names[thread_id] = {
        "name": "New chat",
        "user_id": x_user_id
    }
    save_names(names)
    return {"id": thread_id}


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str, x_user_id: Optional[str] = Header(None)):
    names = load_names()
    if thread_id in names:
        if not is_authorized(thread_id, x_user_id, names):
            raise HTTPException(status_code=403, detail="Not authorized to delete this thread")
        del names[thread_id]
        save_names(names)
    return {"ok": True}


@app.patch("/api/threads/{thread_id}/name")
def rename_thread(thread_id: str, body: dict, x_user_id: Optional[str] = Header(None)):
    names = load_names()
    if thread_id in names:
        if not is_authorized(thread_id, x_user_id, names):
            raise HTTPException(status_code=403, detail="Not authorized to rename this thread")
            
        info = names[thread_id]
        if isinstance(info, dict):
            info["name"] = body.get("name", "New chat")
        else:
            names[thread_id] = {"name": body.get("name", "New chat"), "user_id": x_user_id}
        save_names(names)
    return {"ok": True}


@app.get("/api/threads/{thread_id}/messages")
def get_messages(thread_id: str, x_user_id: Optional[str] = Header(None)):
    names = load_names()
    # Prevent users from guessing thread IDs and reading others' messages
    if thread_id in names and not is_authorized(thread_id, x_user_id, names):
        raise HTTPException(status_code=403, detail="Not authorized to view this thread")

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
async def upload_pdf(thread_id: str, file: UploadFile = File(...), x_user_id: Optional[str] = Header(None)):
    names = load_names()
    if thread_id in names and not is_authorized(thread_id, x_user_id, names):
        raise HTTPException(status_code=403, detail="Not authorized to upload to this thread")

    data = await file.read()
    summary = ingest_pdf(data, thread_id=thread_id, filename=file.filename)
    return summary


@app.post("/api/threads/{thread_id}/chat")
async def chat(thread_id: str, body: dict, x_user_id: Optional[str] = Header(None)):
    user_input = body.get("message", "")
    names = load_names()
    
    # Secure the chat endpoint
    if thread_id in names and not is_authorized(thread_id, x_user_id, names):
        raise HTTPException(status_code=403, detail="Not authorized to chat in this thread")

    # Name generation for new threads
    info = names.get(thread_id)
    if not info or (isinstance(info, dict) and info.get("name") in ["New chat", "New conversation"]):
        words = user_input.strip().split()
        name = " ".join(words[:5]).capitalize() or "New chat"
        names[thread_id] = {
            "name": name,
            "user_id": x_user_id
        }
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