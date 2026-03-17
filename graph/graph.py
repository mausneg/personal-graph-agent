from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

from graph.state import GraphState
from graph.tools import get_tools
from graph.const import AGENT, TOOLS
from graph.nodes.agent import agent_node


async def create_graph(checkpointer):
    builder = StateGraph(state_schema=GraphState)
    builder.add_node(AGENT, agent_node)
    builder.add_node(TOOLS, ToolNode(await get_tools()))

    builder.add_edge(START, AGENT)
    builder.add_conditional_edges(AGENT, tools_condition)
    builder.add_edge(TOOLS, AGENT)

    graph = builder.compile(checkpointer=checkpointer)
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return graph
