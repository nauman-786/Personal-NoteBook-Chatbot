from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LANGGRAPH_MODEL", "openai/gpt-4o-mini"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
