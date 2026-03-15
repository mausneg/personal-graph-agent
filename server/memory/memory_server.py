from langchain_ollama import OllamaEmbeddings
from langgraph.store.postgres import PostgresStore
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import Any
from dotenv import load_dotenv
import psycopg
import os

load_dotenv()


class ToolResponse(BaseModel):
    status_code: int
    data: Any | None


print("memory server starting...")

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
DB_SERVICE = os.getenv("DB_SERVICE", "localhost")
DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_SERVICE}:{POSTGRES_PORT}/{POSTGRES_DB}"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"Connecting to database at {DB_URL}...")
print(f"Using Ollama at {OLLAMA_BASE_URL}")

mcp = FastMCP("memory", port=os.getenv("MEMORY_SERVER_PORT"))
app = mcp.sse_app()

embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model="qwen3-embedding:0.6b",
)
conn = psycopg.connect(DB_URL, autocommit=True, prepare_threshold=0)


def embed_text(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)


store = PostgresStore(conn=conn, index={"embed": embed_text, "dims": 1024})
store.setup()


@mcp.tool()
def inspect_memory(user_id: str, query: str, limit: int) -> ToolResponse:
    """
    Inspect relevant memory based on user question.
    Usee this tool when you don't know exactly what information you need but want to find relevant memories based on a query.

    Args:
        user_id: user identifier
        query:  user question
        limit: num of category retrieved
    Return:
        List of relevant memories or failure message
    """
    try:
        namespace = (user_id, "preferences")
        memories = store.search(namespace, query=query, limit=limit)
        data = "\n".join([f"- Category {m.key}: {m.value}" for m in memories]) 

        if not memories:
            return ToolResponse(status_code=200, data="No data found").model_dump()

        return ToolResponse(status_code=200, data=data).model_dump()
    except Exception as e:
        return ToolResponse(
            status_code=500, data=f"Failed to inspect memory: {e}"
        ).model_dump()


@mcp.tool()
def save_memory(user_id: str, category: str, information: dict) -> ToolResponse:
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
        return ToolResponse(
            status_code=200, data=f"Successfully saved {category} preferences."
        )
    except Exception as e:
        return ToolResponse(
            status_code=500, data=f"Failed to save {category} preferences: {e}"
        )


@mcp.tool()
def retrieve_memory(user_id: str, category: str) -> ToolResponse:
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
        return ToolResponse(status_code=200, data=f"Category {item.key}: {item.value}")
    except Exception as e:
        return ToolResponse(
            status_code=500, data=f"Failed to retrieve category {category}: {e}"
        )
