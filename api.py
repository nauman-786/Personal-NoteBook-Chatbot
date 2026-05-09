from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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

# Smart Persistence for both Conversations and Users
DATA_DIR = "/data" if os.path.exists("/data") else "."
NAMES_FILE = os.path.join(DATA_DIR, "conversation_names.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


# --- AUTHENTICATION LOGIC ---
def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

class AuthRequest(BaseModel):
    username: str
    password: str

@app.post("/api/auth")
def authenticate(req: AuthRequest):
    users = load_users()
    u = req.username.strip()
    p = req.password.strip()

    if not u or not p:
        raise HTTPException(status_code=400, detail="Username and password required")

    if u not in users:
        # Auto-register new users
        users[u] = p
        save_users(users)
        return {"message": "New account created!"}
    else:
        # Check password for existing users
        if users[u] != p:
            raise HTTPException(status_code=401, detail="Incorrect password")
        return {"message": "Welcome back!"}


# --- CHAT LOGIC ---
def load_names() -> dict:
    if os.path.exists(NAMES_FILE):
        try:
            with open(NAMES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_names(names: dict):
    with open(NAMES_FILE, "w") as f:
        json.dump(names, f, indent=2)

def get_thread_info(thread_id: str, names: dict) -> dict:
    info = names.get(thread_id)
    if isinstance(info, dict):
        return info
    elif isinstance(info, str):
        return {"name": info, "user_id": None}
    else:
        return {"name": "New chat", "user_id": None}


@app.get("/api/threads")
def get_threads(x_user_id: Optional[str] = Header(None, alias="X-User-ID")):
    names = load_names()
    result = []
    needs_save = False
    
    for tid_str, info_raw in names.items():
        info = get_thread_info(tid_str, names)
        
        # Auto-claim old chats
        if info.get("user_id") is None and x_user_id:
            info["user_id"] = x_user_id
            names[tid_str] = info
            needs_save = True

        # Multi-Tenant Filter: Show if it belongs to this user
        if not x_user_id or info.get("user_id") == x_user_id:
            result.append({
                "id": tid_str,
                "name": info.get("name", "New chat"),
                "has_document": thread_has_document(tid_str),
                "document": thread_document_metadata(tid_str),
            })
            
    if needs_save:
        save_names(names)
        
    return {"threads": result[::-1]}


@app.post("/api/threads")
def create_thread(x_user_id: Optional[str] = Header(None, alias="X-User-ID")):
    thread_id = str(uuid.uuid4())
    names = load_names()
    names[thread_id] = {
        "name": "New chat",
        "user_id": x_user_id
    }
    save_names(names)
    return {"id": thread_id}


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str, x_user_id: Optional[str] = Header(None, alias="X-User-ID")):
    names = load_names()
    info = get_thread_info(thread_id, names)
    
    if info.get("user_id") and info.get("user_id") != x_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    if thread_id in names:
        del names[thread_id]
        save_names(names)
    return {"ok": True}


@app.patch("/api/threads/{thread_id}/name")
def rename_thread(thread_id: str, body: dict, x_user_id: Optional[str] = Header(None, alias="X-User-ID")):
    names = load_names()
    info = get_thread_info(thread_id, names)
    
    if info.get("user_id") and info.get("user_id") != x_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    info["name"] = body.get("name", "New chat")
    info["user_id"] = info.get("user_id") or x_user_id
    names[thread_id] = info
    save_names(names)
    
    return {"ok": True}


@app.get("/api/threads/{thread_id}/messages")
def get_messages(thread_id: str, x_user_id: Optional[str] = Header(None, alias="X-User-ID")):
    names = load_names()
    info = get_thread_info(thread_id, names)
    
    if info.get("user_id") and info.get("user_id") != x_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

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
async def upload_pdf(thread_id: str, file: UploadFile = File(...), x_user_id: Optional[str] = Header(None, alias="X-User-ID")):
    names = load_names()
    info = get_thread_info(thread_id, names)
    
    if info.get("user_id") and info.get("user_id") != x_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    data = await file.read()
    summary = ingest_pdf(data, thread_id=thread_id, filename=file.filename)
    return summary


@app.post("/api/threads/{thread_id}/chat")
async def chat(thread_id: str, body: dict, x_user_id: Optional[str] = Header(None, alias="X-User-ID")):
    user_input = body.get("message", "")
    names = load_names()
    info = get_thread_info(thread_id, names)
    
    if info.get("user_id") and info.get("user_id") != x_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if info.get("name") in ["New chat", "New conversation"]:
        words = user_input.strip().split()
        info["name"] = " ".join(words[:5]).capitalize() or "New chat"
    
    info["user_id"] = info.get("user_id") or x_user_id
    names[thread_id] = info
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