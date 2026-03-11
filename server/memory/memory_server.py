from langchain_ollama import OllamaEmbeddings
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import Item
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Literal
import psycopg
import os

load_dotenv()

mcp = FastMCP("db", port=8003)
embeddings = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model="qwen3-embedding:0.6b",
)
conn = psycopg.connect(os.getenv("DB_URL"), autocommit=True, prepare_threshold=0)


def embed_text(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)


store = PostgresStore(conn=conn, index={"embed": embed_text, "dims": 1024})
store.setup()


@mcp.tool()
def save_memory(user_id: str, category: str, information: dict) -> str:
    """
    Save user preferences or information for long-term memory.

    Args:
        user_id: user identifier
        category: category of information (e.g. "food", "hobby", "daily")
        information: dictionary containing information to save
    Return:
        Success or failure message
    """

    try:
        namespace = (user_id, "preferences")
        store.put(namespace, category, information)
        return f"Sucessfully saved {category} preferences."
    except Exception as e:
        return f"Failed saved {category} preferences: {e}"


@mcp.tool()
def retrieve_memory(user_id: str, category: str) -> Literal[Item, str]:
    """
    Retrieve user preferences or information for long-term memory.

    Args:
        user_id: user identifier
        category: category of information (e.g. "food", "hobby", "daily")
    Return:
        Item or failure message
    """
    try:
        namespace = (user_id, "preferences")
        item = store.get(namespace, category)
        if not item:
            raise ValueError(f"No {category} preferences found for user {user_id}.")
        return item
    except Exception as e:
        return f"Failed retrieve category {category}: {e}"

if __name__ == "__main__":
    mcp.run(transport="sse")