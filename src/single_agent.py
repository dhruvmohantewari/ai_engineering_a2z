import os
import certifi

import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import Client
from langgraph.prebuilt import create_react_agent
import langchainhub as hub
from urllib3.util import response

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearchResults(max_results=2)
result = search_tool.invoke("give me the latest news on AI")
tools = [search_tool]

llm = ChatOllama(
    model="llama3.1:8b",  # use a name from `ollama list`
    base_url="http://localhost:11434",
    temperature=0,
)

# # one full reply
# print(llm.invoke("Say hello in one sentence.").content)
#
# # streamed, token by token
# for chunk in llm.stream("Name three uses of tmux."):
#     print(chunk.content, end="", flush=True)
# print()

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful assistant. Use your tools whenever math operations are requested.",
)

inputs = {
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of india. when find the current weather in delhi",
        }
    ]
}

print("--- Running Agent ---")
for event in agent.stream(inputs, stream_mode="debug"):
    print(event)
    # for node_name, state_update in chunk.items():
    #     print(f"\n[Node: {node_name}]")
    #     # Pretty printing the state additions (e.g., the prompt, tool calls, and tool outputs)
    #     for msg in state_update.get("messages", []):
    #         msg.pretty_print()
