from langchain_core.messages import HumanMessage
from pprint import pprint
from dotenv import load_dotenv

from graph.chains.generation import generation_chain

load_dotenv()

def test_save_memory_tool_call()-> None:
    messages = HumanMessage("i like to play dota 2 at weekend")
    response =  generation_chain.invoke({
        "user_id": "mausneg",
        "memories": "",
        "messages": [messages]
    })

    pprint(response)
    assert response.tool_calls[0]["name"] == "save_memory"
    
def test_retrieve_memory_tool_call()-> None:
    messages = HumanMessage("what do you know about me?")
    response =  generation_chain.invoke({
        "user_id": "mausneg",
        "memories": "",
        "messages": [messages]
    })

    pprint(response)
    assert response.tool_calls[0]["name"] == "retrieve_memory"