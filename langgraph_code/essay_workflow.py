from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langgraph.pregel.main import Output
from langsmith import prompt_cache
from pydantic import BaseModel, Field
import operator


model = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="1h",
)


class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)


structured_model = model.with_structured_output(EvaluationSchema)

with open("./langgraph_code/essay.txt", "r", encoding="utf-8") as file:
    essay = file.read()

prompt = f"Evaluate the language of the follow essay and provide a feedback assign a score out of 10 \n {essay}"
result = structured_model.invoke(prompt)


class EssayState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_score: Annotated[list[int], operator.add]
    avg_score: float


def evaluate_language(state: EssayState) -> EssayState:
    prompt = f"Evaluate language quality of the following essay and provide a feedback and assign a score out of 10\n {state['essay']}"
    output = structured_model.invoke(prompt)
    return {"language_feedback": output.feedback, "individual_score": [output.score]}


def evaluate_analysis(state: EssayState) -> EssayState:
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10\n {state['essay']}"
    output = structured_model.invoke(prompt)
    return {"analysis_feedback": output.feedback, "individual_score": [output.score]}


def evaluate_thought(state: EssayState) -> EssayState:
    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10\n {state['essay']}"
    output = structured_model.invoke(prompt)
    return {"clarity_feedback": output.feedback, "individual_score": [output.score]}


def final_evaluation(state: EssayState) -> EssayState:
    prompt = f"Based of the following feedbacks create a summarized feedback \n language_feedback - {state['language_feedback']} \n depth of analysis feedback - {state['analysis_feedback']} \n clarity of thought feedback - {state['clarity_feedback']}"
    overall_feedback = model.invoke(prompt).content
    avg_score = sum(state["individual_score"]) / len(state["individual_score"])

    return {"overall_feedback": overall_feedback, "avg_score": avg_score}


graph = StateGraph(EssayState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge(START, "evaluate_language")

graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("evaluate_language", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()

with open("langgraph_code/essay.txt", "r", encoding="utf-8") as file:
    essay = file.read()

initial_state = {"essay": essay}
result = workflow.invoke(initial_state)
print(result)
