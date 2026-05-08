import uuid
import json
import os
import tempfile

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langraph_rag_backend import (
    chatbot,
    retrieve_all_threads,
    ingest_pdf,
    _get_retriever,
    _THREAD_RETRIEVERS,
    _THREAD_METADATA,
)

st.set_page_config(page_title="Personal LLM", page_icon="◉", layout="wide")

# Theme configuration
THEMES = {
    "White": {
        "bg_main": "#ffffff",
        "bg_secondary": "#f7f7f5",
        "surface": "#ffffff",
        "text_primary": "#000000",
        "text_secondary": "#4f4f4f",
        "accent": "#C96A2E",
        "sidebar_bg": "#f7f7f5",
        "user_msg": "#f0f0f0",
        "assistant_text": "#111111",
        "border": "#e6e6e0",
    },
    "Black": {
        "bg_main": "#1A1A1A",
        "bg_secondary": "#111111",
        "surface": "#171717",
        "text_primary": "#ffffff",
        "text_secondary": "#bbbbbb",
        "accent": "#C96A2E",
        "sidebar_bg": "#111111",
        "user_msg": "#2A2A2A",
        "assistant_text": "#f5f5f5",
        "border": "#2b2b2b",
    },
    "Glass": {
        "bg_main": "rgba(255,255,255,0.68)",
        "bg_secondary": "rgba(255,255,255,0.40)",
        "surface": "rgba(255,255,255,0.48)",
        "text_primary": "#111111",
        "text_secondary": "#4b4b4b",
        "accent": "#C96A2E",
        "sidebar_bg": "rgba(255,255,255,0.40)",
        "user_msg": "rgba(240,240,240,0.90)",
        "assistant_text": "#111111",
        "border": "rgba(20,20,20,0.10)",
    },
}

# Load conversation names from file
NAMES_FILE = "conversation_names.json"


def load_conversation_names():
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_conversation_names(names):
    with open(NAMES_FILE, "w") as f:
        json.dump(names, f)


def apply_theme(theme_name):
    theme = THEMES.get(theme_name, THEMES["White"])
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght@400&display=swap');

            :root {{
                --bg-main: {theme['bg_main']};
                --bg-secondary: {theme['bg_secondary']};
                --surface: {theme['surface']};
                --text-primary: {theme['text_primary']};
                --text-secondary: {theme['text_secondary']};
                --accent: {theme['accent']};
                --sidebar-bg: {theme['sidebar_bg']};
                --user-msg: {theme['user_msg']};
                --assistant-text: {theme['assistant_text']};
                --border: {theme['border']};
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                background: var(--bg-main);
                color: var(--text-primary);
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}

            #MainMenu, footer, header {{visibility: hidden;}}

            .block-container {{
                padding-top: 1rem;
                max-width: 760px;
            }}

            section[data-testid="stSidebar"] {{
                background: var(--sidebar-bg) !important;
                border-right: 1px solid var(--border);
            }}

            .material-symbols-outlined {{
                font-family: 'Material Symbols Outlined';
                font-weight: normal;
                font-style: normal;
                font-size: 20px;
                line-height: 1;
                letter-spacing: normal;
                text-transform: none;
                display: inline-block;
                white-space: nowrap;
                direction: ltr;
                -webkit-font-feature-settings: 'liga';
                -webkit-font-smoothing: antialiased;
            }}

            .brand-row {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.5rem 0 1rem;
                color: var(--text-primary);
                font-weight: 650;
                letter-spacing: -0.02em;
            }}

            .brand-icon {{
                width: 34px;
                height: 34px;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: var(--surface);
                display: grid;
                place-items: center;
            }}

            .section-label {{
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: var(--text-secondary);
                margin: 0.9rem 0 0.5rem;
            }}

            .stButton > button {{
                border-radius: 999px;
                border: 1px solid var(--border);
                background: var(--surface);
                color: var(--text-primary);
                box-shadow: none;
            }}

            .stButton > button:hover {{
                border-color: var(--accent);
                color: var(--accent);
            }}

            .stTextInput > div > div > input {{
                border-radius: 999px !important;
                border: 1px solid var(--border) !important;
                background: var(--surface) !important;
                color: var(--text-primary) !important;
            }}

            .top-title {{
                text-align: center;
                font-weight: 650;
                font-size: 1rem;
                color: var(--text-primary);
                letter-spacing: -0.01em;
                margin: 0.3rem 0 1rem;
            }}

            .message-area {{
                padding-bottom: 0.5rem;
            }}

            .user-row {{
                display: flex;
                justify-content: flex-end;
                margin: 0.65rem 0;
            }}

            .user-bubble {{
                max-width: 78%;
                background: var(--user-msg);
                border: 1px solid var(--border);
                color: var(--text-primary);
                border-radius: 20px;
                padding: 0.85rem 1rem;
                font-size: 15px;
                line-height: 1.6;
                white-space: pre-wrap;
            }}

            .assistant-wrap {{
                width: 100%;
                margin: 0.5rem 0 1rem;
            }}

            .assistant-head {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.35rem;
                color: var(--text-secondary);
                font-size: 0.85rem;
            }}

            .assistant-body {{
                color: var(--assistant-text);
                font-size: 15px;
                line-height: 1.7;
            }}

            .glass-panel {{
                background: rgba(255,255,255,0.45);
                border: 1px solid rgba(255,255,255,0.35);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
            }}

            [data-testid="stChatInput"] {{
                background: transparent;
            }}

            .chat-title {{
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -0.04em;
                color: var(--text-primary);
                margin: 0.25rem 0 0.25rem;
            }}

            .chat-subtitle {{
                color: var(--text-secondary);
                margin-bottom: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def delete_chat(thread_id):
    """Delete a chat conversation."""
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)
    if thread_id in st.session_state["conversation_names"]:
        del st.session_state["conversation_names"][thread_id]
        save_conversation_names(st.session_state["conversation_names"])
    # If deleting current chat, switch to new chat
    if st.session_state.get("thread_id") == thread_id:
        reset_chat()
        st.rerun()


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


def ingest_text_file(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    """Ingest a text file with vector embeddings (RAG)."""
    try:
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        
        # Decode text
        text = file_bytes.decode("utf-8", errors="ignore")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        
        # Create document-like objects
        from langchain_core.documents import Document
        docs = [Document(page_content=text, metadata={"source": filename})]
        chunks = splitter.split_documents(docs)
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Store for this thread
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename,
            "documents": 1,
            "chunks": len(chunks),
        }
        
        return {
            "filename": filename,
            "documents": 1,
            "chunks": len(chunks),
        }
    except Exception as e:
        raise ValueError(f"Error ingesting text file: {str(e)}")


def ingest_document(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    """Ingest any document type with RAG (PDF or text)."""
    if filename.lower().endswith(".pdf"):
        return ingest_pdf(file_bytes, str(thread_id), filename)
    elif filename.lower().endswith((".txt", ".md")):
        return ingest_text_file(file_bytes, str(thread_id), filename)
    else:
        raise ValueError("Unsupported file type. Upload PDF or TXT.")


def get_thread_document() -> dict:
    """Get document metadata for current thread."""
    thread_id = st.session_state.get("thread_id")
    if thread_id and str(thread_id) in _THREAD_METADATA:
        return _THREAD_METADATA[str(thread_id)]
    return {}


# Initialize session state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
if "conversation_names" not in st.session_state:
    st.session_state["conversation_names"] = load_conversation_names()
if "theme" not in st.session_state:
    st.session_state["theme"] = "White"
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

add_thread(st.session_state["thread_id"])

# Sidebar controls
st.sidebar.markdown(
    """
    <div class="brand-row">
        <div class="brand-icon"><span class="material-symbols-outlined">psychology</span></div>
        <div>Personal LLM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="section-label">Appearance</div>', unsafe_allow_html=True)
theme_options = list(THEMES.keys())
current_theme = st.session_state["theme"] if st.session_state["theme"] in THEMES else "White"
if current_theme != st.session_state["theme"]:
    st.session_state["theme"] = current_theme
theme = st.selectbox("Theme", theme_options, index=theme_options.index(current_theme), label_visibility="collapsed")
if theme != st.session_state["theme"]:
    st.session_state["theme"] = theme
    st.rerun()

apply_theme(st.session_state["theme"])

st.sidebar.divider()
st.sidebar.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Upload a document (PDF, TXT)",
    type=["pdf", "txt", "md"],
    key="file_uploader",
    label_visibility="collapsed",
)
if uploaded_file:
    if uploaded_file.name not in st.session_state["uploaded_files"]:
        st.session_state["uploaded_files"].append(uploaded_file.name)
        # Ingest document with RAG (vector embeddings)
        try:
            thread_id = str(st.session_state["thread_id"])
            file_bytes = uploaded_file.getvalue()
            metadata = ingest_document(file_bytes, thread_id, uploaded_file.name)
            st.sidebar.success(
                f"✅ {uploaded_file.name}\n📊 {metadata['chunks']} chunks from {metadata['documents']} page(s)"
            )
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    else:
        st.sidebar.info(f"📌 {uploaded_file.name} already indexed")

if st.session_state["uploaded_files"]:
    st.sidebar.write("**Indexed Documents:**")
    for file in st.session_state["uploaded_files"]:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"📎 {file}")
        if col2.button("✕", key=f"del-{file}"):
            st.session_state["uploaded_files"].remove(file)
            st.rerun()

st.sidebar.divider()
st.sidebar.markdown('<div class="section-label">Chats</div>', unsafe_allow_html=True)

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    conv_name = st.session_state["conversation_names"].get(thread_id, f"Chat {thread_id[:8]}")
    
    col1, col2 = st.sidebar.columns([3, 1])
    
    if col1.button(conv_name, key=f"thread-{thread_id}", use_container_width=True):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        st.session_state["message_history"] = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]
        st.rerun()
    
    if col2.button("🗑️", key=f"delete-{thread_id}", help="Delete this chat"):
        delete_chat(thread_id)
        st.rerun()

st.markdown(
    """
    <div class="top-title">Personal LLM</div>
    <div class="chat-subtitle" style="text-align:center;">Simple, polished, and built for focused conversation.</div>
    """,
    unsafe_allow_html=True,
)

# Show current conversation name with edit option
current_name = st.session_state["conversation_names"].get(
    st.session_state["thread_id"], 
    f"Chat {st.session_state['thread_id'][:8]}"
)

col1, col2 = st.columns([3, 1])
with col1:
    new_name = st.text_input(
        "Chat name",
        value=current_name,
        label_visibility="collapsed",
        key="name_input",
        placeholder="Rename this conversation",
    )
    if new_name and new_name != current_name:
        st.session_state["conversation_names"][st.session_state["thread_id"]] = new_name
        save_conversation_names(st.session_state["conversation_names"])
        st.rerun()

with col2:
    if st.button("Rename", use_container_width=True):
        st.session_state["edit_mode"] = not st.session_state.get("edit_mode", False)
        st.rerun()

# Show uploaded documents with metadata
if st.session_state["uploaded_files"]:
    doc_metadata = get_thread_document()
    if doc_metadata:
        with st.expander(f"Indexed document: {doc_metadata.get('filename')}"):
            col1, col2 = st.columns(2)
            col1.metric("Pages", doc_metadata.get("documents", 0))
            col2.metric("Chunks", doc_metadata.get("chunks", 0))
            st.caption("Vector embeddings are ready for semantic retrieval.")
    else:
        st.info(f"{len(st.session_state['uploaded_files'])} document(s) ready.")

st.markdown('<div class="message-area">', unsafe_allow_html=True)
for message in st.session_state["message_history"]:
    if message["role"] == "user":
        st.markdown(f"<div class='user-row'><div class='user-bubble'>{message['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class='assistant-wrap'>
                <div class='assistant-head'>
                    <span class="material-symbols-outlined">smart_toy</span>
                    <span>Personal LLM</span>
                </div>
                <div class='assistant-body'>{message['content']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
st.markdown('</div>', unsafe_allow_html=True)

placeholder = "Message Personal LLM..."
user_input = st.chat_input(placeholder, key="user_input")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    config = {"configurable": {"thread_id": str(st.session_state["thread_id"])} }
    with st.spinner("Thinking..."):
        ai_message = st.write_stream(
            chunk.content
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
            if isinstance(chunk, AIMessage)
        )

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
