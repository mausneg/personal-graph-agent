from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

MEMORY_SERVER_PORT = os.getenv("MEMORY_SERVER_PORT")
MEMORY_SERVER_SERVICE = os.getenv("MEMORY_SERVER_SERVICE", "localhost")


async def get_tools() -> list[BaseTool]:
    client = MultiServerMCPClient(
        {
            "airbnb": {
                "command": "npx",
                "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
                "transport": "stdio",
            },
            "memory": {
                "url": f"http://{MEMORY_SERVER_SERVICE}:{MEMORY_SERVER_PORT}/sse",
                "transport": "sse",
            },
        }
    )

    tools = await client.get_tools()
    return tools
