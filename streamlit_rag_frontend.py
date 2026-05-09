import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

st.set_page_config(page_title="LangGraph PDF Chat", page_icon="📄", layout="centered")
NEW_CHAT_TITLE = "New Chat"

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ececec;
    }

    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #2a2a2a;
    }

    [data-testid="stChatMessage"] {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border: 1px solid #2a2a2a;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #2a2a2a !important;
        border: 1px solid #3a3a3a !important;
        border-radius: 12px !important;
        color: #ececec !important;
    }

    .stButton > button {
        background-color: #2a2a2a;
        color: #ececec;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
    }

    .stButton > button:hover {
        background-color: #3a3a3a;
        border-color: #4a4a4a;
    }

    [data-testid="stSidebar"] .stButton > button {
        text-align: left;
        padding: 8px 12px;
        font-size: 13px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    h1 {
        font-size: 1.5rem !important;
        font-weight: 600;
        color: #ececec;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def generate_thread_id():
    return uuid.uuid4()


def generate_chat_name(user_message: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return NEW_CHAT_TITLE
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Generate a short 4-6 word title for this conversation based on the user's "
                        "first message. Return only the title."
                    )
                ),
                HumanMessage(content=user_message),
            ]
        )
        title = (response.content or "").strip()
        return title or NEW_CHAT_TITLE
    except Exception:
        return NEW_CHAT_TITLE


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["chat_names"].setdefault(str(thread_id), NEW_CHAT_TITLE)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    st.session_state["chat_names"].setdefault(str(thread_id), NEW_CHAT_TITLE)


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
if "chat_names" not in st.session_state:
    st.session_state["chat_names"] = {}
if "stop_requested" not in st.session_state:
    st.session_state["stop_requested"] = False
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if "renaming_thread" not in st.session_state:
    st.session_state["renaming_thread"] = None

add_thread(st.session_state["thread_id"])
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Current Chat:** {st.session_state['chat_names'].get(thread_key, NEW_CHAT_TITLE)}")
st.sidebar.caption(f"`{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` ({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(uploaded_pdf.getvalue(), thread_id=thread_key, filename=uploaded_pdf.name)
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
st.sidebar.text_input("🔍 Search chats...", key="search_query")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        thread_id_key = str(thread_id)
        name = st.session_state["chat_names"].get(thread_id_key, NEW_CHAT_TITLE)
        search_query = st.session_state.get("search_query", "").strip().lower()
        if search_query and search_query not in name.lower():
            continue

        row_col1, row_col2, row_col3 = st.sidebar.columns([6, 1, 1])
        with row_col1:
            if st.button(name, key=f"side-thread-{thread_id}", use_container_width=True):
                selected_thread = thread_id
        with row_col2:
            if st.button("✏️", key=f"rename-thread-{thread_id}"):
                st.session_state["renaming_thread"] = thread_id_key
                st.rerun()
        with row_col3:
            if st.button("🗑️", key=f"delete-thread-{thread_id}"):
                st.session_state["chat_threads"] = [tid for tid in st.session_state["chat_threads"] if str(tid) != thread_id_key]
                st.session_state["chat_names"].pop(thread_id_key, None)
                st.session_state["ingested_docs"].pop(thread_id_key, None)
                if str(st.session_state["thread_id"]) == thread_id_key:
                    reset_chat()
                st.rerun()

        if st.session_state.get("renaming_thread") == thread_id_key:
            rename_key = f"rename-input-{thread_id}"
            st.sidebar.text_input("Rename chat", value=name, key=rename_key)
            save_col, cancel_col = st.sidebar.columns([1, 1])
            with save_col:
                if st.button("Save", key=f"save-rename-{thread_id}", use_container_width=True):
                    new_name = st.session_state.get(rename_key, "").strip()
                    if new_name:
                        st.session_state["chat_names"][thread_id_key] = new_name
                    st.session_state["renaming_thread"] = None
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key=f"cancel-rename-{thread_id}", use_container_width=True):
                    st.session_state["renaming_thread"] = None
                    st.rerun()

st.title("Multi Utility Chatbot")

if st.session_state["is_generating"]:
    if st.button("⏹ Stop generating"):
        st.session_state["stop_requested"] = True

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    has_no_prior_user_messages = not any(msg["role"] == "user" for msg in st.session_state["message_history"])
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    if has_no_prior_user_messages and st.session_state["chat_names"].get(thread_key, NEW_CHAT_TITLE) == NEW_CHAT_TITLE:
        st.session_state["chat_names"][thread_key] = generate_chat_name(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    status_holder = {"box": None}
    st.session_state["is_generating"] = True
    st.session_state["stop_requested"] = False

    def ai_only_stream():
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if st.session_state.get("stop_requested", False):
                break
            if isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "tool")
                if status_holder["box"] is None:
                    status_holder["box"] = st.status(f"Using {tool_name}", expanded=True)
                else:
                    status_holder["box"].update(label=f"Using {tool_name}", state="running", expanded=True)
            if isinstance(chunk, AIMessage):
                yield chunk.content

    with st.chat_message("assistant"):
        ai_message = st.write_stream(ai_only_stream())
    st.session_state["is_generating"] = False

    if status_holder["box"] is not None:
        status_holder["box"].update(label="Tool finished", state="complete", expanded=False)

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} (chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.session_state["renaming_thread"] = None
    messages = load_conversation(selected_thread)
    st.session_state["message_history"] = [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
        for msg in messages
    ]
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
