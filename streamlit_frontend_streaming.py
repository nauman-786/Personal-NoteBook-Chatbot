import streamlit as st
from langchain_core.messages import HumanMessage

from langgraph_backend import chatbot

st.set_page_config(page_title="LangGraph Streaming Chat", page_icon="💬", layout="centered")

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

st.title("LangGraph Streaming Chat")
st.caption("Token streaming demo")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            chunk.content
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
            if getattr(chunk, "content", None)
        )

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
