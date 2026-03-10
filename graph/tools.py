from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

tools = [TavilySearch()]