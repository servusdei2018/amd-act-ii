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

## Running the Stack (Server and Router)

Execute the unified startup command:
```bash
./run_server.sh
```

Triduum Router will serve an OpenAI-compatible API on port `8001`. You can integrate it with any client, SDK, or Web UI that supports OpenAI endpoints by configuring the API base URL to `http://localhost:8001/v1`.

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

## Running via Container (Docker/Podman)

A pre-configured `Dockerfile` is provided to package and deploy the entire stack:

```bash
podman build -t triduum-stack:latest .

podman run --name triduum-test -d \
  -p 8001:8001 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  triduum-stack:latest
```


