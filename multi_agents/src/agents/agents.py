from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()


llm = ChatOllama(
    model="llama3.1:8b",  # use a name from `ollama list`
    base_url="http://localhost:11434",
    temperature=0,
    keep_alive="1h",
)


# 1st agents
def build_search_agent():
    return create_agent(model=llm, tools=[web_search])


def build_reader_agent():
    return create_agent(model=llm, tools=[scrape_url])


writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are an expert research writer. Write clear, structured insightful report.",
        ),
        (
            "human",
            """write a detailed reaseach report on the topic below
     Topic: {topic}
     Research Gathered:
     {research}

     Structure the report as:
     - Introduction
     - Key Findings (minimum 3 well-explained points)
     - Conclusion
     - Sources (list all URLs found in the research)

     Be detailed, factual and professional.
     """,
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()

# critic_chain

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a sharp and constructive research critic. Be honest and specific.",
        ),
        (
            "human",
            """Review the research report below and evaluate it strictly.

    Report:
    {report}

    Respond in this exact format:

    Score: X/10

    Strengths:
     - ...
     - ...

    Areas to Improve:
     - ...
     - ...

    One line verdict:
     ...""",
        ),
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()
