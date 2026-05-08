---
title: Personal Notebook Chatbot
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_rag_frontend.py
pinned: false
---

# Chatbot in LangGraph

A Streamlit + LangGraph chatbot project with multiple variants:

- basic chat
- streaming chat
- threaded chat
- tool-using chat
- SQLite-backed chat history
- PDF RAG chat
- MCP-backed chat

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with at least:

```env
OPENAI_API_KEY=your_key_here
```

Optional extras used by some demos:

```env
ALPHAVANTAGE_API_KEY=your_key_here
```

## Run

Basic chat:

```bash
streamlit run streamlit_frontend.py
```

Other demos:

```bash
streamlit run streamlit_frontend_streaming.py
streamlit run streamlit_frontend_threading.py
streamlit run streamlit_frontend_tool.py
streamlit run streamlit_frontend_database.py
streamlit run streamlit_rag_frontend.py
streamlit run streamlit_frontend_mcp.py
```

## Notes

- The MCP demo expects reachable MCP servers if you configure them.
- The RAG demo indexes PDF files per thread and stores retrieval state in memory.
- The database and tool demos use a local `chatbot.db` SQLite file for persistence.