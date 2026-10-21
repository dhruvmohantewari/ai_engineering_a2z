from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import operator

generator_llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="10m",
)

evaluator_llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="10m",
)

optimizer_llm = ChatOllama(
    model="qwen3.5:9b",
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="10m",
)


class PostEvalution(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        description="Final evaluation result."
    )
    feedback: str = Field(..., description="feedback for the facebook post")


str_eval_llm = evaluator_llm.with_structured_output(PostEvalution)


class PostState(TypedDict):
    topic: str
    post: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    interation: int
    max_interation: int

    post_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


def generate_post(state: PostState):
    messages = [
        SystemMessage(content="You are a funny and clever facebook influencer."),
        HumanMessage(
            content=f"""
        write a short, original and hilarious facebook post on the topic: '{state["topic"]}'.

        Rules:
        - Do NOT use question-answer format.
        - Max 500 characters.
        - Use observational humor, irony, sarcam or cultural references.
        - Think in meme logic, punchlines or relatable takes.
        - use simple day to day english"""
        ),
    ]

    response = generator_llm.invoke(messages).content
    return {"post": response, "post_history": [response]}


def evaluate_post(state: PostState):
    messages = [
        SystemMessage(
            content="You are a ruthless, no-laugh-given facebook critic. You evaluate posts based on humor, originality, virality and post format"
        ),
        HumanMessage(
            content=f"""
        Evaluate the following facebook post:

        Post: '{state["post"]}'

        Use the criteria below to evaluate

        1. Originality - Is this fresh or have you seen it hundred of times before?
        2. Humor - Did it genuinely make you smile, laugh or chuckle?
        3. Punches - Is it short, sharp and scroll-stopping?
        4. Virality Potential - Would people share, react or comment on it?
        5. Format - Is it a well-formated facebook post (not a setup punchline joke, not a Q&A joke and under 500 characters)?

        Auto-reject if:
        - It's written in a Q&A format (e.g., 'why did you...' or 'what happens when..'
        - It exceeds 500 characters
        - It reads like a traditional setup punchline joke
        - Dont end with generic, throwaway or deflating lines that weakens the humor (e.g., "Masterpices of the auntie-uncle universe" or vauge summary)
        
        ### Respond only in structured format:
        - evaluation: 'approved', 'needs_improvement'
        - feedback: One paragraph explaining strengths and weaknes
        """
        ),
    ]

    response = str_eval_llm.invoke(messages)
    return {"evaluation": response.evaluation, "feedback": response.feedback}


def optimize_post(state: PostState):
    messages = [
        SystemMessage(
            content="you punch up facebook post for virality and humor based on given feeback."
        ),
        HumanMessage(
            content=f"""
        Improve the facebook post based on this feedback:
        {state["feedback"]}

        topic: {state["topic"]}
        Original post:
        {state["post"]}

        Re-write it as a short, viral-worthy facebook post. Avoid Q&A style and stay under 500 characters.
        """
        ),
    ]

    response = optimizer_llm.invoke(messages).content
    iteration = state["iteration"] + 1

    return {"post": response, "interation": interation, "post_history": [response]}


def route_evaluation(state: PostState):
    if (
        state["evaluation"] == "approved"
        or state["interation"] >= state["max_interation"]
    ):
        return "approved"
    else:
        return "needs_improvement"


graph = StateGraph(PostState)
graph.add_node("generate", generate_post)
graph.add_node("evaluate", evaluate_post)
graph.add_node("optimize", optimize_post)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges(
    "evaluate", route_evaluation, {"approved": END, "needs_improvement": "optimize"}
)
graph.add_edge("optimize", "evaluate")

workflow = graph.compile()

inital_state = {"topic": "agentic AI", "iteration": 1, "max_interation": 5}
result = workflow.invoke(inital_state)
print(result)
