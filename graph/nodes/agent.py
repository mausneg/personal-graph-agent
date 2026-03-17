from dotenv import load_dotenv

from graph.state import GraphState
from graph.chains.generation import get_generation_chain

load_dotenv()


async def agent_node(state: GraphState) -> GraphState:
    generation_chain = await get_generation_chain()

    messages = state["messages"]
    user_id = state["user_id"]

    response = generation_chain.invoke({"user_id": user_id, "messages": messages})

    return {"messages": [response]}
