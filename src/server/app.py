import json
from typing import Any, AsyncGenerator
import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse

from src.config import VLLM_BASE_URL, AGENT_MODEL_NAME, BACKEND_MODEL
from src.logging_config import logger
from src.agent.harness import AgenticHarness

app = FastAPI(title="Triduum Agent Gateway", version="0.1.0")
harness = AgenticHarness()
http_client = httpx.AsyncClient(base_url=VLLM_BASE_URL)


@app.get("/health")
@app.get("/v1/health")
async def health_check() -> dict[str, Any]:
    """Exposes gateway and backend connectivity health metrics."""
    backend_status = "unreachable"
    available_models = []

    try:
        response = await http_client.get("/models", timeout=2.0)
        if response.status_code == 200:
            backend_status = "ok"
            data = response.json()
            available_models = [m["id"] for m in data.get("data", [])]
    except Exception as e:
        logger.warning(f"Backend connectivity check failed: {e}")

    return {
        "gateway_status": "ok",
        "backend_status": backend_status,
        "backend_models": available_models,
        "configured_agent": AGENT_MODEL_NAME,
        "configured_backend_model": BACKEND_MODEL,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Exposes available models, merging backend options with our Triduum agent."""
    models = [
        {
            "id": AGENT_MODEL_NAME,
            "object": "model",
            "created": 1719878400,
            "owned_by": "agentic-harness",
        }
    ]

    try:
        response = await http_client.get("/models", timeout=2.0)
        if response.status_code == 200:
            backend_data = response.json()
            for model in backend_data.get("data", []):
                # Avoid duplicates
                if model["id"] != AGENT_MODEL_NAME:
                    models.append(model)
    except Exception as e:
        logger.warning(f"Could not fetch models from backend vLLM: {e}")

    return {"object": "list", "data": models}


async def stream_agent_response(
    generator: AsyncGenerator[Any, None],
) -> AsyncGenerator[str, None]:
    """Formats the harness stream generator into standard SSE content lines."""
    try:
        async for chunk in generator:
            # chunk is a ChatCompletionChunk object from openai library
            # Serialize chunk to JSON string and yield as event stream
            yield f"data: {chunk.model_dump_json()}\n\n"
    except Exception as e:
        logger.error(f"Error during agentic stream generation: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """Handles chat completions, routing to the agentic harness or proxying to vLLM."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model")
    stream = body.get("stream", False)

    if not model:
        raise HTTPException(status_code=400, detail="Missing required field 'model'")

    # Route to Triduum
    if model == AGENT_MODEL_NAME:
        messages = body.get("messages", [])
        # Extract extra OpenAI arguments to pass down
        kwargs = {
            k: v for k, v in body.items() if k not in ["model", "messages", "stream"]
        }

        if stream:
            gen = harness.generate_stream(messages, **kwargs)
            return StreamingResponse(
                stream_agent_response(gen), media_type="text/event-stream"
            )
        else:
            completion = await harness.generate_completion(messages, **kwargs)
            return Response(
                content=completion.model_dump_json(), media_type="application/json"
            )

    # Proxy all other models to backend vLLM server
    else:
        logger.info(
            f"[Proxy] Routing request for model '{model}' directly to backend vLLM..."
        )
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ["host", "content-length"]
        }

        if stream:

            async def proxy_stream() -> AsyncGenerator[bytes, None]:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{VLLM_BASE_URL}/chat/completions",
                        json=body,
                        headers=headers,
                        timeout=60.0,
                    ) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk

            return StreamingResponse(proxy_stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{VLLM_BASE_URL}/chat/completions",
                        json=body,
                        headers=headers,
                        timeout=60.0,
                    )
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        media_type=response.headers.get(
                            "content-type", "application/json"
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to proxy completion request to backend: {e}")
                    raise HTTPException(
                        status_code=502, detail=f"Bad Gateway proxying to vLLM: {e}"
                    )
