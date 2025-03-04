from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# For ChatBot
import phi.api
import phi
from phi.playground import Playground, serve_playground_app



import os
from dotenv import load_dotenv
load_dotenv()

os.environ["PHI_API_KEY"] = os.getenv("PHI_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_id = Groq(id="deepseek-r1-distill-llama-70b")
# model_id = Groq(id="llama-3.3-70b-versatile")

# Web-Based Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=model_id,
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information in the answer"],
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

app = Playground(agents=[web_search_agent, financial_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True) # Filename : app ( line no : 46)