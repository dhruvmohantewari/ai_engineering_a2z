from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages


llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434", temperature=0)


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": [response]}


checkpoint = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chatnode", chat_node)

graph.add_edge(START, "chatnode")
graph.add_edge("chatnode", END)

chatbot = graph.compile(checkpointer=checkpoint)
