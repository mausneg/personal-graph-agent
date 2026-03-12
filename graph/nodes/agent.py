from dotenv import load_dotenv

from graph.state import GraphState
from graph.chains.generation import generation_chain

load_dotenv()

def agent_node(state: GraphState)-> GraphState:
    messages = state["messages"]
    user_id = state["user_id"]
    
    response = generation_chain.invoke({
        "user_id": user_id,
        "memories": "",
        "messages": messages
    })
    
    return {"messages": [response]}