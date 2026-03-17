import pytest
from langchain_core.messages import HumanMessage
from pprint import pprint
from dotenv import load_dotenv

from graph.chains.generation import get_generation_chain

load_dotenv()


@pytest.mark.asyncio
async def test_save_memory_tool_call() -> None:
    messages = HumanMessage("i like to play dota 2 at weekend")
    generation_chain = await get_generation_chain()
    response = generation_chain.invoke({"user_id": "mausneg", "messages": [messages]})

    pprint(response)
    assert response.tool_calls[0]["name"] == "save_memory"


@pytest.mark.asyncio
async def test_retrieve_memory_tool_call() -> None:
    messages = HumanMessage("what do you know about my daily activity?")
    generation_chain = await get_generation_chain()
    response = generation_chain.invoke({"user_id": "mausneg", "messages": [messages]})

    pprint(response)
    assert response.tool_calls[0]["name"] == "retrieve_memory"


@pytest.mark.asyncio
async def test_inspect_memory_tool_call() -> None:
    messages = HumanMessage("give me summary about my self")
    generation_chain = await get_generation_chain()
    response = generation_chain.invoke({"user_id": "mausneg", "messages": [messages]})

    pprint(response)
    assert response.tool_calls[0]["name"] == "inspect_memory"
