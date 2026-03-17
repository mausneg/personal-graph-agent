from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage
from datetime import datetime
from dotenv import load_dotenv
import psycopg
import os

from graph.graph import create_graph

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
DB_SERVICE = os.getenv("DB_SERVICE", "localhost")
DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_SERVICE}:{POSTGRES_PORT}/{POSTGRES_DB}"


async def chat_with_llm(query: str, user_id: str, session_id: str) -> str:
    async with await psycopg.AsyncConnection.connect(
        DB_URL, autocommit=True, prepare_threshold=0
    ) as conn:
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()

        graph = await create_graph(checkpointer)
        response = await graph.ainvoke(
            {"messages": [HumanMessage(query)], "user_id": user_id},
            config={"configurable": {"thread_id": session_id}},
        )

        return response["messages"][-1].content


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health_check():
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/chat")
async def chat(
    query: str = Form(...), user_id: str = Form(...), session_id: str = Form(...)
):
    start_time = datetime.now()
    answer = await chat_with_llm(query, user_id, session_id)
    inference_time = (datetime.now() - start_time).total_seconds()
    return JSONResponse(
        status_code=200, content={"answer": answer, "inference_time": inference_time}
    )
