from __future__ import annotations

import asyncio
import os
import threading
from typing import Annotated, TypedDict

import aiosqlite
import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception:
    MultiServerMCPClient = None

load_dotenv()

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    return _submit_async(coro)


llm = ChatOpenAI(
    model=os.getenv("LANGGRAPH_MODEL", "openai/gpt-4o-mini"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    response = requests.get(url, timeout=30)
    return response.json()


MCP_SERVERS: dict[str, dict[str, object]] = {}
client = None
if MultiServerMCPClient is not None and MCP_SERVERS:
    client = MultiServerMCPClient(MCP_SERVERS)


def load_mcp_tools() -> list[BaseTool]:
    if client is None:
        return []
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()
tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def chat_node(state: ChatState):
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

tool_node = ToolNode(tools) if tools else None

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node is not None:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
