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

## Verifying Server Status

Once the server is running on `http://localhost:8000`, verify the connection and model routing with the asynchronous client orchestrator:

```bash
uv run main.py
```
