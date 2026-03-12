from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage

from graph.state import GraphState
from graph.tools import tool_node
from graph.const import AGENT, TOOLS
from graph.nodes.agent import agent_node

builder = StateGraph(context_schema=GraphState)
builder.add_node(AGENT, agent_node)
builder.add_node(TOOLS, tool_node)

builder.add_edge(START, AGENT)
builder.add_conditional_edges(AGENT, tools_condition)

graph = builder.compile()