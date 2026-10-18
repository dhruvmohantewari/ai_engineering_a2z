from langgraph.graph import StateGraph, START, END, state
from typing import TypedDict


class EmployeeState(TypedDict):
    employee_name: str
    monthly_salary: int
    working_days: int
    completed_project: int

    yearly_salary: int
    bonus_amount: int
    project_status: str
    summary: str


def calculate_yearly_salary(state: EmployeeState) -> EmployeeState:
    yearly_salary = state["monthly_salary"] * 12
    # state["yearly_salary"] = yearly_salary
    return {"yearly_salary": yearly_salary}


def calculate_bonus(state: EmployeeState) -> EmployeeState:
    bonus_amount = state["monthly_salary"] * 2
    # state["bonus_amount"] = bonus_amount
    return {"bonus_amount": bonus_amount}


def project_evaluation(state: EmployeeState) -> EmployeeState:
    if state["completed_project"] >= 5:
        status = "Excellent"
    else:
        status = "Average"

    # state["project_status"] = status
    return {"project_status": status}


def summary(state: EmployeeState) -> EmployeeState:
    summary_text = f"Employee {state['employee_name']} has a yearly salary of {state['yearly_salary']} and bonus of {state['bonus_amount']}. Project status is {state['project_status']}."
    # state["summary"] = summary_text
    return {"summary": summary_text}


graph = StateGraph(EmployeeState)

graph.add_node("calculate_bonus", calculate_bonus)
graph.add_node("calculate_yearly_salary", calculate_yearly_salary)
graph.add_node("project_evaluation", project_evaluation)
graph.add_node("summary", summary)

graph.add_edge(START, "calculate_bonus")
graph.add_edge(START, "project_evaluation")
graph.add_edge(START, "calculate_yearly_salary")

graph.add_edge("calculate_yearly_salary", "summary")
graph.add_edge("calculate_bonus", "summary")
graph.add_edge("project_evaluation", "summary")

graph.add_edge("summary", END)

workflow = graph.compile()
print(workflow)
input = {
    "employee_name": "Sam",
    "monthly_salary": 200,
    "working_days": 26,
    "completed_project": 7,
}

final_state = workflow.invoke(input)
print(final_state)
