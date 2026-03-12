from langchain_ollama import OllamaEmbeddings
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import Item, SearchItem
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
def inspect_memory(user_id: str, query: str)-> Literal[list[SearchItem], str]:
    """
    Inspect relevant memory based on user question.
    Usee this tool when you don't know exactly what information you need but want to find relevant memories based on a query.
    
    Args:
        user_id: user identifier
        query:  user question
    Return:
        List of relevant memories or failure message
    """
    try:
        namespace = (user_id, "preferences")
        memories = store.search(namespace, query=query, limit=3)
        if not memories:
            return "No relevant memories found."
        return memories
    except Exception as e:
        return f"Failed to inspect memory: {e}"    

@mcp.tool()
def save_memory(user_id: str, category: str, information: dict) -> str:
    """
    Save user preferences or information for long-term memory.
    Always use this tool to save new information or update existing information in the user's memory. 
    The category can be used to organize different types of information (e.g. "food", "hobby", "daily").

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
        return f"Successfully saved {category} preferences."
    except Exception as e:
        return f"Failed to save {category} preferences: {e}"


@mcp.tool()
def retrieve_memory(user_id: str, category: str) -> Literal[Item, str]:
    """
    Retrieve user preferences or information for long-term memory.
    Use this tool when you know the specific category of information you want to retrieve.

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