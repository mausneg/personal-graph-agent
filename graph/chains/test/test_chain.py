from langchain_core.messages import ToolMessage
from pprint import pprint
from dotenv import load_dotenv

from graph.chains.generation import generation_chain

load_dotenv()

def test_save_memory_tool_call()-> None:
    question = "i like to play dota 2 at weekend"
    result =  generation_chain.invoke({
        "user_id": "mausneg",
        "memories": "",
        "question": question
    })

    pprint(result)
    assert result.tool_calls[0]["name"] == "save_memory"
    
def test_retrieve_memory_tool_call()-> None:
    question = "what do you know about me?"
    result =  generation_chain.invoke({
        "user_id": "mausneg",
        "memories": "",
        "question": question
    })

    pprint(result)
    assert result.tool_calls[0]["name"] == "retrieve_memory"