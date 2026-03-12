from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

from graph.tools import tools

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            """
            You are a helpful assistant with long-term memory capabilities and access to utility tools.

            User ID:
            {user_id}

            Current User Memories:
            {memories}

            Available Tools:
            {tools}

            Guidelines:
            - Always save when user shares personal information
            - Retrieve specific categories when needed for context
            - Be conversational and natural when using all tools
            - Combine tools when appropriate
            """
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(tools="\n".join(f"{t.name}: {t.description}"for t in tools))


llm_with_tools = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), 
    model="qwen3:4b"
    ).bind_tools(tools)

generation_chain = prompt | llm_with_tools