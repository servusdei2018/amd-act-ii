# AMD Act II Hackathon: Local Gemma Agent Pipeline

My submission for the AMD Act II hackathon. Hybrid Token-Efficient Routing Agent track.

## Setup and Dependency Locking

The dependency management for this project is powered by `uv` for lightning-fast and reproducible builds.

### 1. Lock the Dependencies

```bash
uv lock
```

### 2. Synchronize the Environment

```bash
uv sync --frozen
```

## Running the vLLM Server

The `run_server.sh` script automates environment isolation, applies driver overrides required for RDNA3 hardware, and starts the OpenAI-compatible endpoint.

Execute the server command:
```bash
./run_server.sh
```

This launches the `cyankiwi/gemma-4-31B-it-AWQ-4bit` model utilizing AWQ 4-bit quantization, optimized to consume up to 90% GPU memory allocation.

## Running the Triduum Router

Once the server is running on `http://localhost:8000`, verify the connection and model routing with the asynchronous client orchestrator:

```bash
uv run main.py
```

> [!NOTE]
> Since the Triduum Router serves an OpenAI-compatible API on port `8001`, you can integrate it with any client, SDK, or Web UI that supports OpenAI endpoints (by configuring the API base URL to `http://localhost:8001/v1`).

## Running the CLI Chat Client

To interact with the Triduum Router gateway through a terminal interface, run the included interactive chat client:

```bash
uv run cli.py
```

The interactive chat client provides a real-time streaming chat experience and supports several slash commands:
* `/info` — Display current configuration, active model, and session statistics.
* `/system <prompt>` — Update the system prompt (this will clear and reset session history).
* `/clear` or `/reset` — Clear the active conversation history.
* `/quit` or `/exit` — Exit the client.

