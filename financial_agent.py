from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["PHI_API_KEY"] = os.getenv("PHI_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# model_id = Groq(id="deepseek-r1-distill-llama-70b")
model_id = Groq(id="llama-3.3-70b-versatile")

# Web-Based Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=model_id,
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,
)

# Financial Agent
financial_agent = Agent(
    name="Financial AI Agent",
    model=model_id,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to show the data"],
    show_tool_calls=True,
    markdown=True,
)

# Multi Agent
multi_ai_agent = Agent(
    team=[web_search_agent, financial_agent],
    model=model_id,
    instructions=["Always include the sources","Use tables to show the data"],
    show_tool_calls=True,
    markdown=True,
)

# multi_ai_agent.print_response("Summarize analyst recommendationof NVDA,share the latest news for NVDA",stream=True)
multi_ai_agent.print_response("What is the stock price of JioMart?What are the analyst recommendations for JioMart?",stream=True)