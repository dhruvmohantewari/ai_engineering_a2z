from langgraph.cache import base
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from ollama import ResponseError

llm = ChatOllama(
    model='llama3.2:3b',
    base_url='http://localhost:11434',
    temperature=0,
    keep_alive='5m'
)

class JokeState(TypedDict):
    topic: str
    joke: str
    explaination: str

def generate_joke(state: JokeState):
    prompt = f'Generate a joke on the topic {state['topic']}'
    response = llm.invoke(prompt).content

    return {'joke': response}

def generate_explanation(state: JokeState):
    prompt = f'write an explation for the joke {state['joke'}'
    response = llm.invoke(prompt).content

    return {'explaination': response}

graph = StateGraph(JokeState)
graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke','generate_explanation')
graph.add_edge('generate_explanation', END)

checkpoint = InMemorySaver()
workflow = graph.compile(checkpointer=checkpoint)

config1 = {''}
