# Personal Graph Agent

Personal Graph Agent is a self-experiment project to explore how an AI assistant can combine:

- short-term memory (conversation thread)
- long-term memory (user preferences/knowledge)
- external tools (MCP tools)

The current implementation uses LangGraph + Ollama + Postgres (with pgvector) and exposes a simple chat interface with Streamlit.

## Why I Built This

This project is part of my personal portfolio and experiment track to understand:

- how to design stateful AI agents
- how to persist memory across sessions/users
- how to orchestrate tool calls in a graph workflow
- trade-offs between local/self-hosted AI stack and developer experience

## Key Features

- Graph-based agent orchestration with LangGraph.
- Short-term memory using thread-based checkpointing in Postgres.
- Long-term memory server using MCP + vector search (pgvector).
- Tool integration:
	- memory tools (`save_memory`, `retrieve_memory`, `inspect_memory`)
	- Airbnb MCP tool server (`@openbnb/mcp-server-airbnb`)
- FastAPI chat backend endpoint (`/chat`).
- Streamlit chat frontend for interactive testing.

## High-Level Architecture

```text
User (Streamlit UI)
				|
				v
Agent API (FastAPI + LangGraph)
				|
				+--> Ollama (LLM + embeddings)
				|
				+--> Postgres (checkpoint + pgvector memory store)
				|
				+--> MCP Tools
						 +--> Memory Server (custom)
						 +--> Airbnb MCP Server
```

Main services from `docker-compose.yml`:

- `database`: Postgres + pgvector
- `ollama`: local LLM/embedding runtime
- `memory_server`: MCP server for long-term memory operations
- `agent`: FastAPI + LangGraph orchestration
- `view`: Streamlit frontend

## Tech Stack

- Python 3.10
- FastAPI
- LangChain + LangGraph
- LangChain MCP Adapters
- Ollama (`qwen3.5:4b`, `qwen3-embedding:0.6b`)
- PostgreSQL 15 + pgvector
- Streamlit
- Docker Compose

## Repository Structure

```text
.
|- docker-compose.yml
|- graph/
|  |- app.py                # FastAPI app for agent
|  |- graph.py              # LangGraph definition
|  |- chains/generation.py  # Prompt + model/tool binding
|  |- nodes/agent.py        # Agent node execution
|  |- tools.py              # MCP tool wiring
|  |- test/                 # Graph behavior tests
|- server/memory/
|  |- memory_server.py      # MCP memory server
|- views/
|  |- chat.py               # Streamlit chat UI
|- database/
|  |- Dockerfile            # Postgres + pgvector image
```

## Environment Variables

Create a `.env` file in the project root.

Required keys used by this project:

```env
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=
LANGCHAIN_ENDPOINT=
LANGCHAIN_PROJECT=
TAVILY_API_KEY=

POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_PORT=
POSTGRES_DB=

MEMORY_SERVER_PORT=
AGENT_PORT=
```

Notes:

- `LANGCHAIN_*` and `TAVILY_API_KEY` are optional depending on your tracing/search usage.
- Keep `POSTGRES_PORT`, `MEMORY_SERVER_PORT`, and `AGENT_PORT` consistent with your runtime setup.

## Quick Start (Docker)

1. Build and run all services.

```bash
docker compose up --build -d
```

2. Pull required Ollama models.

```bash
docker compose exec ollama ollama pull qwen3.5:4b
docker compose exec ollama ollama pull qwen3-embedding:0.6b
```

3. Verify health endpoint for agent.

```bash
curl http://localhost:${AGENT_PORT}/health
```

4. Open chat UI.

By default, `view` service does not publish a host port in current compose file. You can either:

- run Streamlit locally (recommended for development), or
- add a `ports` mapping to `view` service (for example `8502:8502`) and restart compose.

## Run UI Locally (Optional)

After backend services are running:

```bash
export AGENT_SERVICE=localhost
export AGENT_PORT=<your_agent_port>
streamlit run views/chat.py --server.port 8502
```

Then open: `http://localhost:8502`

## API Usage

Sample chat request:

```bash
curl -X POST http://localhost:${AGENT_PORT}/chat \
	-F "query=hello, my name is Surya" \
	-F "user_id=surya" \
	-F "session_id=session-001"
```

Response shape:

```json
{
	"answer": "...",
	"inference_time": 1.23
}
```

## Running Tests

Tests exist for chain/tool-call behavior and memory flow:

```bash
pytest graph/chains/test/test_chain.py
pytest graph/test/test_graph.py
```

Make sure dependent services (Postgres, memory server, ollama) are available before running integration-heavy tests.

## Known Limitations

- Model response formatting can be inconsistent depending on prompt/model behavior.
- First inference may be slow if models are not warmed up.
- Airbnb MCP output can be verbose; additional response post-processing may be needed for cleaner UX.
- Current setup is experimental and not production-hardened.

## Portfolio Notes

If you are reviewing this project as part of my portfolio, this repository demonstrates:

- end-to-end AI system design (UI + API + agent graph + memory server)
- local-first AI stack orchestration with Docker
- practical use of MCP tools in agent workflows
- experiment-driven engineering with test coverage for core behavior

## Future Improvements

- Add structured response formatter layer (summary/cards/table normalization).
- Add stronger guardrails for tool usage and output formatting.
- Add observability dashboard (latency/tool-call tracing).
- Improve deployment profile for cloud and non-GPU environments.