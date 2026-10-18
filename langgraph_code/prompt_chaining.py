from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict

model = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="1m",
)


class BlogState(TypedDict):
    title: str
    outline: str
    content: str


def create_outline(state: BlogState) -> BlogState:
    title = state["title"]

    prompt = f"Generate a detailed outline for a blog on the topic - {title}"
    outline = model.invoke(prompt).content

    state["outline"] = outline

    return state


def create_blog(state: BlogState) -> BlogState:
    title = state["title"]
    outline = state["outline"]

    prompt = f"Write a detailed blog on the tile - {title} using the following outline {outline}"
    content = model.invoke(prompt).content

    state["content"] = content

    return state


graph = StateGraph(BlogState)
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)

workflow = graph.compile()

input = {"title": "sustainability on planet Mars"}
final_state = workflow.invoke(input)
print(final_state)
