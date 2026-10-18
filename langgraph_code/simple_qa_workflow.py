from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="1h",
)


class LLMState(TypedDict):
    question: str
    answer: str


def llm_qa(state: LLMState) -> LLMState:
    question = state["question"]

    prompt = f"answer following question {question}"

    answer = model.invoke(prompt).content

    state["answer"] = answer

    return state


graph = StateGraph(LLMState)
graph.add_node("llm_qa", llm_qa)

graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

workflow = graph.compile()

input = {"question": "who is the creator of python"}
final_state = workflow.invoke(input)
print(final_state)
