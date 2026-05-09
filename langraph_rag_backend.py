from __future__ import annotations

import os
import sqlite3
import io
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LANGGRAPH_MODEL", "openai/gpt-4o-mini"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    try:
        # Read the PDF directly from memory to bypass Hugging Face disk restrictions
        pdf_stream = io.BytesIO(file_bytes)
        pdf_reader = PdfReader(pdf_stream)

        docs = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename or "Uploaded PDF", "page": i}
                ))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or "Uploaded PDF",
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return {
            "filename": filename or "Uploaded PDF",
            "documents": len(docs),
            "chunks": len(chunks),
        }
    except Exception as e:
        print(f"Error ingesting PDF: {e}")
        raise e


search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations (add, sub, mul, div) on two numbers."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a given stock symbol using Alpha Vantage API."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    response = requests.get(url, timeout=30)
    return response.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve the most relevant document chunks for a query from the active chat's indexed RAG store."""
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # FIX 1: Dynamically check if a document is currently loaded for this chat
    has_doc = thread_id and str(thread_id) in _THREAD_RETRIEVERS
    doc_info = _THREAD_METADATA.get(str(thread_id), {})
    
    # Inject state-awareness into the AI's prompt so it knows the file is there
    if has_doc:
        filename = doc_info.get("filename", "the document")
        doc_context = (
            f"CRITICAL STATE: The user has uploaded a document named '{filename}'. "
            f"You MUST use the `rag_tool` with thread_id `{thread_id}` to search its contents "
            "before answering any questions about it."
        )
    else:
        doc_context = "No document is currently uploaded. If the user asks to check a document, ask them to upload it."

    system_message = SystemMessage(
        content=(
            f"You are a helpful assistant.\n{doc_context}\n"
            "You can also use web search, stock price, and calculator tools when helpful."
        )
    )
    
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


# FIX 2: Smart Persistence. 
# If HF persistent storage is enabled, it uses it. Otherwise, it safely defaults to local.
DB_PATH = "/data/chatbot.db" if os.path.exists("/data") else "chatbot.db"
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads() -> list[str]:
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})