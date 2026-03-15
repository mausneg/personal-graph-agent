from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pprint import pprint
import uuid
import psycopg
import os
import pytest

from graph.graph import create_graph

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
DB_SERVICE = os.getenv("DB_SERVICE", "localhost")
DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_SERVICE}:{POSTGRES_PORT}/{POSTGRES_DB}"


@pytest.mark.asyncio
async def test_short_term_memory_answer_yes() -> None:
    conn = await psycopg.AsyncConnection.connect(
        DB_URL, autocommit=True, prepare_threshold=0
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = create_graph(checkpointer)

    response = await graph.ainvoke(
        {"user_id": thread_id, "messages": [HumanMessage("hello, my name is mausneg")]},
        config=config,
    )
    print(f"[RESPONSE]: {response['messages'][-1].content}")

    response = await graph.ainvoke(
        {
            "user_id": thread_id,
            "messages": [HumanMessage("what's my name? (yes/no answer only)")],
        },
        config=config,
    )
    answer = response["messages"][-1].content
    print(f"[ANSWER]: {answer}")

    await conn.close()

    assert "yes" in answer.lower()


@pytest.mark.asyncio
async def test_short_term_memory_answer_no() -> None:
    conn = await psycopg.AsyncConnection.connect(
        DB_URL, autocommit=True, prepare_threshold=0
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = create_graph(checkpointer)

    response = await graph.ainvoke(
        {
            "user_id": thread_id,
            "messages": [HumanMessage("what's my name? (yes/no answer only)")],
        },
        config=config,
    )
    answer = response["messages"][-1].content
    print(f"[ANSWER]: {answer}")

    await conn.close()

    assert "no" in answer.lower()


@pytest.mark.asyncio
async def test_long_term_memory_save() -> None:
    conn = await psycopg.AsyncConnection.connect(
        DB_URL, autocommit=True, prepare_threshold=0
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = create_graph(checkpointer)

    response = await graph.ainvoke(
        {
            "user_id": "mausneg",
            "messages": [HumanMessage("i like to play dota 2 at weekend")],
        },
        config=config,
    )
    await conn.close()

    pprint(response)

@pytest.mark.asyncio
async def test_long_term_retrieve() -> None:
    conn = await psycopg.AsyncConnection.connect(
        DB_URL, autocommit=True, prepare_threshold=0
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = create_graph(checkpointer)

    response = await graph.ainvoke(
        {
            "user_id": "mausneg",
            "messages": [HumanMessage("what do you know about my daily activity?")],
        },
        config=config,
    )
    await conn.close()

    pprint(response)
    
@pytest.mark.asyncio
async def test_long_term_inspect() -> None:
    conn = await psycopg.AsyncConnection.connect(
        DB_URL, autocommit=True, prepare_threshold=0
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = create_graph(checkpointer)

    response = await graph.ainvoke(
        {
            "user_id": "mausneg",
            "messages": [HumanMessage("what do you know about me?")],
        },
        config=config,
    )
    await conn.close()

    pprint(response)