# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SuperBizAgent (UI: 智能OnCall助手) is a FastAPI + LangChain/LangGraph service with two independent agent flows:

- **RAG chat** (`/api/chat`, `/api/chat_stream`): answers questions from the Markdown runbooks in `aiops-docs/`, indexed into Milvus.
- **AIOps diagnosis** (`/api/aiops`): a Plan-Execute-Replan graph that calls log/monitoring tools over MCP and streams a Markdown report.

LLM calls go through DashScope via `langchain_qwq.ChatQwen`; embeddings use DashScope `text-embedding-v4` (1024 dimensions); the vector store is Milvus; the frontend is vanilla JS in `static/`. Code comments, docstrings, prompts, and UI strings are Chinese — keep that style.

`PROJECT_GUIDE.md` is a long, line-cited tour of the codebase (architecture, request flows, known doc drift). Read it for deeper context; this file is the operational summary.

## Commands

Run commands from the repository root: `.env`, `static/`, `uploads/`, `logs/`, and `volumes/` are all resolved relative to the current working directory.

### Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
# edit .env and set a real DASHSCOPE_API_KEY
```

The Makefile hardcodes `.venv/bin/python`; without a `.venv`, service targets fail.

### Run

```bash
make init       # first run: Milvus stack -> all services -> wait -> upload aiops-docs
make dev        # foreground uvicorn with reload on 0.0.0.0:9900
make run        # foreground uvicorn without reload
make start      # background: CLS MCP :8003, Monitor MCP :8004, FastAPI :9900
make stop       # stop FastAPI + both MCP services
make restart
make up/down/status    # Milvus stack only (docker compose -f vector-database.yml)
make check             # curl http://localhost:9900/health
make status-mcp        # check both MCP services
make upload            # POST every aiops-docs/*.md to /api/upload
make test-upload       # upload one file and print the response
```

Ports: FastAPI/Web UI `9900` (API docs at `/docs`), CLS MCP `8003`, Monitor MCP `8004`, Milvus `19530`, Attu `8000`, MinIO `9000`/`9001`. `make init` requires Docker (or Colima) and a valid key; `make start` assumes Milvus is already up. On Windows use `start-windows.bat` / `stop-windows.bat`; the Makefile is POSIX-oriented.

Debugging:

```bash
make check                     # curl /health
make logs                      # tail server.log
tail -f logs/app_$(date +%Y-%m-%d).log   # Loguru file log
tail -f mcp_cls.log mcp_monitor.log     # MCP services
make status-mcp
```

### Quality

```bash
make format        # ruff isort fix + ruff format / black
make lint
make fix
make type-check    # mypy app/
make security      # bandit
make pre-commit-install
make pre-commit
make test          # currently fails: pyproject sets testpaths=["tests"], but tests/ does not exist
```

`make security` (bandit) and `make watch` (pytest-watch) depend on packages that are not in the `[dev]` extra and fail on a fresh dev install; `make lint`/`format`/`type-check` work with the declared extras.

Once a `tests/` directory exists, run a single test with:

```bash
.venv/bin/python -m pytest tests/test_file.py::test_name -q
```

## Architecture

```text
static/app.js -> app/main.py -> app/api/{chat,aiops,file,health}.py
                                  |- services/rag_agent_service.py -> tools + agent/mcp_client.py -> MCP servers
                                  |- services/aiops_service.py -> agent/aiops/{planner,executor,replanner,state}.py
                                  |- services/vector_index_service.py -> document_splitter -> vector_store_manager -> Milvus
                                  `- core/milvus_client.py
```

Routers are thin; services own the logic. Everything is a module-level singleton (`config`, `rag_agent_service`, `aiops_service`, `milvus_manager`, `vector_store_manager`, `vector_embedding_service`, `vector_index_service`, `document_splitter_service`) — there is no DI container.

### HTTP surface

| Route | Purpose |
|---|---|
| `POST /api/chat` | Non-streaming RAG answer |
| `POST /api/chat_stream` | SSE RAG answer (`content` / `done` / `error`) |
| `POST /api/chat/clear` | Delete the in-memory session |
| `GET /api/chat/session/{session_id}` | Read session history |
| `POST /api/aiops` | SSE Plan-Execute-Replan diagnosis |
| `POST /api/upload` | Save and index one `txt`/`md` file |
| `POST /api/index_directory` | Index every `txt`/`md` in a directory |
| `GET /health` | Liveness; returns 503 when Milvus is unavailable |

### RAG chat

- `RagAgentService` builds a LangChain `create_agent` over `ChatQwen(model=config.rag_model)` with local tools (`retrieve_knowledge`, `get_current_time`) plus dynamically loaded MCP tools.
- Lazy initialization on the first request: `get_mcp_client_with_retry().get_tools()`. This call has no fallback — an exception fails the first chat request, so start both MCP services first. Whether one unreachable server raises or returns partial tools depends on `langchain-mcp-adapters` (not runtime-verified).
- `MemorySaver` checkpoints use `thread_id = session_id`. Sessions are in-process and disappear on restart; the app assumes a single worker.
- `query_stream()` emits only `AIMessage` text blocks. `api/chat.py` maps them to SSE `{"type":"content"|"done"|"error"}`.
- `trim_messages_middleware` exists but is never registered, so history grows unbounded.

### AIOps diagnosis

- `AIOpsService.diagnose()` wraps a fixed Chinese diagnosis prompt (asks for a Markdown alert report), then runs a compiled `StateGraph`: `planner -> executor -> replanner`, with a conditional edge back to the executor.
- Planner: retrieves knowledge docs first, lists all tools, then uses `with_structured_output(Plan)`; on failure it falls back to a 3-step plan.
- Executor: takes only `plan[0]`, calls tools through `ToolNode`, then makes one final LLM call. It performs at most one round of tool calls per step.
- Replanner: chooses `continue` / `replan` / `respond`. Hard limits: `>= 8` past steps force a final report; `>= 5` past steps forbid `replan`; replan steps are truncated to the number of remaining steps.
- `past_steps` uses `operator.add`; reusing a `session_id` across diagnoses may accumulate steps and hit the limits earlier (LangGraph semantics, not runtime-verified).
- SSE event types: `plan`, `step_complete`, `status`, `report`, `complete`, `error`.

### MCP tools

- `mcp_servers/cls_server.py`: FastMCP `streamable-http` at `127.0.0.1:8003/mcp`; tools `get_current_timestamp`, `get_region_code_by_name`, `get_topic_info_by_name`, `search_topic_by_service_name`, `search_log`.
- `mcp_servers/monitor_server.py`: `127.0.0.1:8004/mcp`; tools `query_cpu_metrics`, `query_memory_metrics`.
- Both return mock data. The tool list in `mcp_servers/README.md` is outdated.
- `app/agent/mcp_client.py` keeps one `MultiServerMCPClient`; `retry_interceptor` retries 3 times with 1s/2s backoff and returns `CallToolResult(isError=True)` instead of raising when exhausted.

### RAG indexing pipeline

- `POST /api/upload` accepts only `txt`/`md`, max 10 MB, saves to `uploads/`, then calls `VectorIndexService.index_single_file`. Indexing errors are logged but the endpoint still returns HTTP 200, so verify in `server.log` / by querying.
- Indexing: delete old chunks where `metadata["_source"] == <absolute posix path>` -> split -> embed -> `add_documents` with UUID ids. Deletion happens before the new write, so a failed re-index leaves that source missing from Milvus.
- Splitter: Markdown by `#`/`##`, then `RecursiveCharacterTextSplitter(chunk_size = CHUNK_MAX_SIZE * 2 = 1600, overlap = 100)`; chunks under 300 chars are merged. Metadata includes `_source`, `_extension`, `_file_name`, `h1`/`h2`.
- Milvus collection `biz`: `id VARCHAR(100) PK`, `vector FLOAT_VECTOR(1024)`, `content VARCHAR(8000)`, `metadata JSON`; index `IVF_FLAT`, metric `L2`.
- Two access paths must stay in sync: `app/core/milvus_client.py` (pymilvus, schema/index) and `app/services/vector_store_manager.py` (`langchain_milvus.Milvus` with `text_field="content"`, `vector_field="vector"`, `metadata_field="metadata"`).
- If an existing collection's vector dimension is not 1024, `milvus_manager.connect()` drops and recreates it, losing existing vectors.

### Configuration

- `app/config.py` uses pydantic-settings and reads `.env` from the current directory; unknown keys are ignored (`extra="ignore"`).
- Every live LLM call uses `config.rag_model` (default `qwen-max`). `DASHSCOPE_MODEL` is only read by the unused `LLMFactory`, so changing it has no effect.
- `DASHSCOPE_API_BASE` is present in `.env` but is not a `Settings` field, so it never reaches the process environment. `ChatQwen` only sees it if exported in the shell (upstream behavior; not verified against the locked version); embeddings hardcode `https://dashscope.aliyuncs.com/compatible-mode/v1`.
- If `.env` still contains the placeholder `DASHSCOPE_API_KEY=your-api-key-here`, the app fails at import; replace it before starting.
- Importing `app.services.vector_embedding_service` raises `ValueError` when `DASHSCOPE_API_KEY` is empty or the literal `your-api-key-here`, so a valid key is required just to import the app.
- `app/core/llm_factory.py` (`ChatOpenAI`), `app/services/vector_search_service.py`, and `app/models/document.py` are unused legacy code.

## Gotchas

- Health endpoint is `GET /health`, not the `/api/health` advertised in the README.
- `make start-api` binds `127.0.0.1`; `make dev`/`make run` bind `0.0.0.0`. Use the latter for LAN access.
- The frontend hardcodes `http://localhost:9900/api` in `static/app.js`.
- Startup requires Milvus: `lifespan` calls `milvus_manager.connect()` without a fallback. Upstream `langchain_milvus.Milvus` is also reported to connect at construction, which may move the failure earlier to import time (not runtime-verified here).
- `/api/chat` returns HTTP 200 with `{"code": 500}` in the body on failure; `/api/chat/clear` and session lookups use real HTTP 500s. Check the body, not just the status.
- Upload and index run synchronously inside async endpoints and can block the event loop on large files or slow embedding calls.
- There is no `tests/`, CI, or Dockerfile; `make test` fails until `tests/` exists.
- README drift: Python requirement is 3.11–3.13 (`.python-version` is 3.13), not 3.10+; MCP target names in `mcp_servers/README.md` are stale; `/api/aiops` uses a fixed diagnosis task rather than fetching alerts.
- No auth, CORS `allow_origins=["*"]`, no auth on the MCP servers, and `/api/index_directory?directory_path=...` indexes any readable directory. Keep the service local.
- `.gitignore` excludes `.claude/` but not `.env`, and local settings files can contain plaintext provider tokens. If this directory becomes a git repository, add `.env` to `.gitignore` before committing.

## Verification status

Confirmed by reading the code, config, and scripts in this repo:

- Routes, ports, Milvus schema/index, splitter sizes, Agent/Graph wiring, MCP tool names, retry behavior, step limits, and configuration resolution.
- At analysis time `.env` still held the literal placeholder key, so no runtime check was performed.
- `tests/`, CI, and a Dockerfile are absent; `make test` fails.

Still needs a runtime check (static inference only):

- Exact behavior of `MultiServerMCPClient.get_tools()` when one MCP server is down.
- Whether `langchain_milvus.Milvus` connects during construction, and whether Milvus being down fails at import rather than in `lifespan`.
- Whether the locked `ChatQwen` version honors a shell-exported `DASHSCOPE_API_BASE`.
- Whether `past_steps` accumulates across repeated `/api/aiops` calls with the same `session_id`.
- `uv` install/sync under Python 3.13 and the full `make init` flow.
