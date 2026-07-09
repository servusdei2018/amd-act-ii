# AMD Act II Hackathon: Local Gemma Agent Pipeline

My submission for the AMD Act II hackathon. Unicorn track.

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

You can run Triduum Router either by pulling the prebuilt container image from the GitHub Container Registry or by building it locally.

### Option A: Pull the Prebuilt Image (Recommended)

You can pull the official prebuilt image directly from GHCR:

```bash
podman pull ghcr.io/servusdei2018/amd-act-ii:latest
```

Then run the container:

```bash
podman run --name triduum-router -d \
  -p 8001:8001 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  ghcr.io/servusdei2018/amd-act-ii:latest
```

### Option B: Build the Image Locally

If you make modifications to the source code, you can build the image locally:

```bash
podman build -t triduum-stack:latest .
```

Then run the locally built image:

```bash
podman run --name triduum-router -d \
  -p 8001:8001 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  triduum-stack:latest
```


