from os import stat

from langgraph import graph
from langgraph.graph import StateGraph, START, END, state
from langchain_ollama import ChatOllama
from typing import TypedDict, Literal
from pydantic import BaseModel, Field

model = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="5m",
)


class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="sentiment of the review"
    )


class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of issue mentioned in the review"
    )
    tone: Literal["angry", "frustrated", "disapointed", "calm"] = Field(
        description="The emotional tone expressed by the user"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How usrgent or critial the issue appears to be"
    )


structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)


class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


def find_sentiment(state: ReviewState):
    prompt = f"For the following review find out the sentiment \n {state['review']}"
    sentiment = structured_model.invoke(prompt).sentiment
    return {"sentiment": sentiment}


def positive_response(state: ReviewState):
    prompt = f'''write a warm thank-you message response to this review:
    \n\n"{state["review"]}\"\n 
    also, kindly ask the user to leave feedback on our website.'''

    response = model.invoke(prompt).content
    return {"response": response}


def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose this negative review:\n\n{state["review"]}\n"
    "return issue_type, tone and urgency."""

    response = structured_model2.invoke(prompt)
    return {"diagnosis": response.model_dump()}


def negative_response(state: ReviewState):
    diagnosis = state["diagnosis"]
    prompt = f"""You are a support assistant. 
    the user has a '{diagnosis["issue_type"]}' issue, sounded '{diagnosis["tone"]}', and marked urgency as '{diagnosis["urgency"]}'.
    write an empathetic, helpful resolution message."""

    response = model.invoke(prompt).content
    return {"response": response}


def check_sentiment(
    state: ReviewState,
) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"


graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)
graph.add_node("positive_response", positive_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)

graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

initial_state = {
    "review": """Disappointing Experience – Needs Significant Improvement

My experience with this software has been quite frustrating. The interface is not as intuitive as expected, performance can be inconsistent, and some features do not work as smoothly as they should. Frequent issues disrupt the workflow and make even basic tasks more time-consuming than necessary.

The overall user experience falls short of expectations, and the software needs better stability, improved usability, and faster, more reliable support. For a technology product, reliability and ease of use should be fundamental priorities.

I hope the team takes this feedback seriously and focuses on fixing the recurring issues. Until there are meaningful improvements, I would hesitate to recommend this software to others.
"""
}

final_state = workflow.invoke(initial_state)
print(final_state)
