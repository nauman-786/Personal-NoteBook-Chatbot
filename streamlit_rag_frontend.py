import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

st.set_page_config(
    page_title="Notebook Chatbot",
    page_icon="📒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main background */
.stApp {
    background: #0d0e11;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2028 !important;
}

[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* Sidebar title */
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #f0f0f0;
    letter-spacing: -0.5px;
    margin-bottom: 0.25rem;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #555;
    margin-bottom: 1.5rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Thread ID pill */
.thread-pill {
    background: #1a1c22;
    border: 1px solid #252830;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    color: #666;
    font-family: 'DM Mono', monospace;
    word-break: break-all;
    margin-bottom: 1rem;
}

/* New chat button */
.stButton > button {
    background: #1e2028 !important;
    color: #c8c8c8 !important;
    border: 1px solid #2a2d38 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #252830 !important;
    border-color: #3a3d4a !important;
    color: #ffffff !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1c22;
    border: 1px dashed #2a2d38;
    border-radius: 10px;
    padding: 0.5rem;
}

[data-testid="stFileUploader"] label {
    color: #888 !important;
    font-size: 13px !important;
}

/* Success box */
.stSuccess {
    background: #0d1f17 !important;
    border: 1px solid #1a3d28 !important;
    border-radius: 8px !important;
    color: #4ade80 !important;
    font-size: 12px !important;
}

/* Info box */
.stInfo {
    background: #111318 !important;
    border: 1px solid #1e2028 !important;
    border-radius: 8px !important;
    color: #666 !important;
    font-size: 12px !important;
}

/* Main chat title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #f0f0f0;
    letter-spacing: -1px;
    margin-bottom: 0.25rem;
}

.main-subtitle {
    font-size: 13px;
    color: #444;
    margin-bottom: 2rem;
    letter-spacing: 0.3px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.75rem 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #1a1c22 !important;
}

/* Chat message text */
[data-testid="stChatMessage"] p {
    color: #c8c8c8 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

/* Avatar */
[data-testid="stChatMessageAvatar"] {
    background: #1e2028 !important;
    border: 1px solid #2a2d38 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #111318 !important;
    border: 1px solid #1e2028 !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #d0d0d0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #444 !important;
}

/* Divider */
hr {
    border-color: #1e2028 !important;
    margin: 1rem 0 !important;
}

/* Status widget */
[data-testid="stStatus"] {
    background: #111318 !important;
    border: 1px solid #1e2028 !important;
    border-radius: 8px !important;
    color: #888 !important;
    font-size: 12px !important;
}

/* Caption */
.stCaption {
    color: #3a3d4a !important;
    font-size: 11px !important;
}

/* Past conversations label */
.past-conv-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #333;
    margin: 1rem 0 0.5rem 0;
    font-weight: 500;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2028; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2a2d38; }
</style>
""", unsafe_allow_html=True)


def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">📒 Notebook</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Personal AI Chatbot</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="thread-pill">🔗 {thread_key[:24]}…</div>', unsafe_allow_html=True)

    if st.button("＋  New chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.markdown("---")

    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.success(f"📄 **{latest_doc.get('filename')}**\n\n{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} pages")
    else:
        st.info("No PDF uploaded yet.")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.info(f"`{uploaded_pdf.name}` already indexed.")
        else:
            with st.status("Indexing PDF…", expanded=True) as status_box:
                summary = ingest_pdf(uploaded_pdf.getvalue(), thread_id=thread_key, filename=uploaded_pdf.name)
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label="✓ PDF ready", state="complete", expanded=False)

    st.markdown("---")
    st.markdown('<div class="past-conv-label">Past conversations</div>', unsafe_allow_html=True)

    if not threads:
        st.markdown('<span style="font-size:12px; color:#333;">No history yet.</span>', unsafe_allow_html=True)
    else:
        for thread_id in threads:
            short_id = str(thread_id)[:16] + "…"
            if st.button(short_id, key=f"side-thread-{thread_id}", use_container_width=True):
                selected_thread = thread_id

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Personal Notebook Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Chat · Search · PDF RAG · Calculator · Stock Prices</div>', unsafe_allow_html=True)

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask anything — or upload a PDF and ask about it…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    status_holder = {"box": None}

    def ai_only_stream():
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "tool")
                if status_holder["box"] is None:
                    status_holder["box"] = st.status(f"⚙ Using {tool_name}…", expanded=True)
                else:
                    status_holder["box"].update(label=f"⚙ Using {tool_name}…", state="running", expanded=True)
            if isinstance(chunk, AIMessage):
                yield chunk.content

    with st.chat_message("assistant"):
        ai_message = st.write_stream(ai_only_stream())

    if status_holder["box"] is not None:
        status_holder["box"].update(label="✓ Done", state="complete", expanded=False)

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(f"📎 {doc_meta.get('filename')} · {doc_meta.get('chunks')} chunks · {doc_meta.get('documents')} pages")

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)
    st.session_state["message_history"] = [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
        for msg in messages
    ]
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()