from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define State


class TemperatureState(TypedDict):
    temp_celsius: float
    temp_fahrenheit: float
    weather_status: str


def convert_temp(state: TemperatureState) -> TemperatureState:
    celsius = state["temp_celsius"]
    fahrenheit = (celsius * 9 / 5) + 32
    state["temp_fahrenheit"] = round(fahrenheit, 2)
    return state


def label_weather(state: TemperatureState) -> TemperatureState:
    fahrenheit = state["temp_fahrenheit"]
    if fahrenheit < 50:
        state["weather_status"] = "Cold"
    elif 50 < fahrenheit < 77:
        state["weather_status"] = "Mild"
    elif 77 < fahrenheit < 95:
        state["weather_status"] = "Hot"
    else:
        state["weather_status"] = "Extreme Heat"

    return state


# Define and compile graph

graph = StateGraph(TemperatureState)

graph.add_node("convert_temp", convert_temp)
graph.add_node("label_weather", label_weather)


graph.add_edge(START, "convert_temp")
graph.add_edge("convert_temp", "label_weather")
graph.add_edge("label_weather", END)

workflow = graph.compile()

initial_sate = {"temp_celsius": 28.5}
final_sate = workflow.invoke(initial_sate)
print(final_sate)
