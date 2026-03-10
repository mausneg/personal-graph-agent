from langchain_ollama import OllamaEmbeddings
from langgraph.store.postgres import PostgresStore
from dotenv import load_dotenv
import psycopg
import os

load_dotenv()

embeddings = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model="qwen3-embedding:0.6b"    
)
conn = psycopg.connect(os.getenv("DB_URL"), autocommit=True, prepare_threshold=0)

def embed_text(texts: list[str])-> list[list[float]]:
    return embeddings.embed_documents(texts) 

store = PostgresStore(conn=conn, index={"embed": embed_text, "dims": 1024})
store.setup()