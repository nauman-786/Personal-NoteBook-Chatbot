# 🤖 Personal Notebook Chatbot

> A modular AI chatbot suite built with **LangGraph** + **Streamlit** — featuring streaming, memory, RAG, tool use, and MCP support.

---

## ✨ Features

| Variant | File | Description |
|---|---|---|
| 🗨️ Basic Chat | `streamlit_frontend.py` | Simple conversational chatbot |
| ⚡ Streaming Chat | `streamlit_frontend_streaming.py` | Token-by-token streaming responses |
| 🧵 Threaded Chat | `streamlit_frontend_threading.py` | Multi-threaded conversation handling |
| 🔧 Tool-Using Chat | `streamlit_frontend_tool.py` | Chatbot with access to external tools |
| 🗄️ Database Chat | `streamlit_frontend_database.py` | Persistent chat history via SQLite |
| 📄 RAG Chat | `streamlit_rag_frontend.py` | PDF-aware chatbot using retrieval-augmented generation |
| 🌐 MCP Chat | `streamlit_frontend_mcp.py` | Chatbot backed by MCP servers |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/nauman-786/Personal-NoteBook-Chatbot.git
cd Personal-NoteBook-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
# Required
OPENAI_API_KEY=your_openai_key_here

# Optional — needed for tool-using demo
ALPHAVANTAGE_API_KEY=your_alphavantage_key_here
```

> ⚠️ **Never commit your `.env` file.** It's already in `.gitignore`.

---

## ▶️ Running the Demos

```bash
# Basic chat
streamlit run streamlit_frontend.py

# Streaming responses
streamlit run streamlit_frontend_streaming.py

# Threaded conversation
streamlit run streamlit_frontend_threading.py

# Tool-using agent
streamlit run streamlit_frontend_tool.py

# SQLite-backed persistent history
streamlit run streamlit_frontend_database.py

# PDF RAG chatbot
streamlit run streamlit_rag_frontend.py

# MCP-backed chatbot
streamlit run streamlit_frontend_mcp.py
```

---

## 🏗️ Project Structure

```
Personal-NoteBook-Chatbot/
│
├── streamlit_frontend.py               # Basic chat UI
├── streamlit_frontend_streaming.py     # Streaming chat UI
├── streamlit_frontend_threading.py     # Threaded chat UI
├── streamlit_frontend_tool.py          # Tool-using chat UI
├── streamlit_frontend_database.py      # Database-backed chat UI
├── streamlit_rag_frontend.py           # RAG chat UI
├── streamlit_frontend_mcp.py          # MCP chat UI
│
├── langgraph_backend.py                # Core LangGraph logic
├── langgraph_database_backend.py       # SQLite persistence backend
├── langgraph_mcp_backend.py            # MCP integration backend
├── langraph_rag_backend.py             # RAG pipeline backend
│
├── conversation_names.json             # Conversation metadata
├── requirements.txt                    # Python dependencies
├── .env                                # API keys (not committed)
└── .gitignore
```

---

## 📝 Notes

- **RAG demo** — indexes PDF files per conversation thread and stores retrieval state in memory. Upload a PDF through the UI to get started.
- **Database demo** — creates a local `chatbot.db` SQLite file automatically on first run.
- **Tool demo** — uses the `ALPHAVANTAGE_API_KEY` to fetch live stock data. Add it to your `.env` to enable.
- **MCP demo** — requires reachable MCP servers. Configure server URLs before running.

---

## 🛠️ Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent orchestration
- [Streamlit](https://streamlit.io) — frontend UI
- [OpenAI](https://platform.openai.com) — LLM backend
- [SQLite](https://www.sqlite.org) — lightweight chat persistence
- [FAISS](https://github.com/facebookresearch/faiss) — vector store for RAG

---

## 📄 License

MIT © [nauman-786](https://github.com/nauman-786)
