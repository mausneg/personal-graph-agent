from langchain_core.tools import tool, BaseTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langgraph.store.base import Item
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from typing import Literal
import asyncio
import os

load_dotenv()

MEMORY_SERVER_PORT = os.getenv("MEMORY_SERVER_PORT")


async def get_tools() -> list[BaseTool]:
    client = MultiServerMCPClient(
        {
            "airbnb": {
                "command": "npx",
                "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
                "transport": "stdio",
            },
            "memory": {
                "url": f"http://127.0.0.1:{MEMORY_SERVER_PORT}/sse",
                "transport": "sse",
            },
        }
    )

    tools = await client.get_tools()
    return tools


tools = asyncio.run(get_tools())
tool_node = ToolNode(tools)

# print(len(tools))
# print(tools)
