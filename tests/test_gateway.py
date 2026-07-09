import pytest
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionMessageParam

from src.config import AGENT_MODEL_NAME, BACKEND_MODEL
from src.agent.harness import AgenticHarness
from src.server.app import app

client = TestClient(app)


def test_config() -> None:
    """Verify default config properties."""
    assert AGENT_MODEL_NAME == "triduum"
    assert BACKEND_MODEL == "cyankiwi/gemma-4-31B-it-AWQ-4bit"


def make_mock_chunk(
    content: str, top_logprobs_pairs: list[tuple[float, float]] | None = None
) -> MagicMock:
    chunk = MagicMock()
    choice = MagicMock()
    choice.delta = MagicMock()
    choice.delta.content = content

    if top_logprobs_pairs is not None:
        choice.logprobs = MagicMock()
        token_logprobs = []
        for p1, p2 in top_logprobs_pairs:
            tl = MagicMock()
            tl.token = content
            tl.logprob = p1

            top1 = MagicMock()
            top1.logprob = p1

            top2 = MagicMock()
            top2.logprob = p2

            tl.top_logprobs = [top1, top2]
            token_logprobs.append(tl)
        choice.logprobs.content = token_logprobs
    else:
        choice.logprobs = None

    chunk.choices = [choice]
    return chunk


@pytest.mark.asyncio
async def test_agentic_harness_certain() -> None:
    """Verify that when the probe is certain, we exit instantly with the probe's answer."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    chunks = [
        make_mock_chunk("Certain", [(0.0, -2.0)]),
        make_mock_chunk(" answer", [(0.0, -2.5)]),
    ]

    async def mock_stream(*args: Any, **kwargs: Any) -> Any:
        async def gen() -> AsyncGenerator[Any, None]:
            for c in chunks:
                yield c

        return gen()

    mock_completions.create.side_effect = mock_stream

    harness = AgenticHarness(client=mock_openai)
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]

    result = await harness.generate_completion(messages=messages)

    assert mock_completions.create.call_count == 1
    kwargs = mock_completions.create.call_args.kwargs
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 2
    assert result.model == AGENT_MODEL_NAME
    assert result.choices[0].message.content == "Certain answer"


@pytest.mark.asyncio
async def test_agentic_harness_hesitant() -> None:
    """Verify that when the probe is hesitant, we run parallel forks and consensus resolver."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    # First token has low margin (0.0 - -1.0 = 1.0 <= 1.5)
    probe_chunks = [
        make_mock_chunk("Hesitant", [(0.0, -1.0)]),
        make_mock_chunk(" draft", [(0.0, -1.0)]),
    ]

    fork1_choice = MagicMock()
    fork1_choice.message.content = "Fork A content"
    fork1_resp = MagicMock()
    fork1_resp.choices = [fork1_choice]

    fork2_choice = MagicMock()
    fork2_choice.message.content = "Fork B content"
    fork2_resp = MagicMock()
    fork2_resp.choices = [fork2_choice]

    fork3_choice = MagicMock()
    fork3_choice.message.content = "Fork C content"
    fork3_resp = MagicMock()
    fork3_resp.choices = [fork3_choice]

    consensus_choice = MagicMock()
    consensus_choice.message.content = "Consensus answer"
    consensus_resp = MagicMock()
    consensus_resp.choices = [consensus_choice]
    consensus_resp.model = BACKEND_MODEL

    static_responses = [fork1_resp, fork2_resp, fork3_resp, consensus_resp]
    static_iter = iter(static_responses)

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):

            async def gen() -> AsyncGenerator[Any, None]:
                for c in probe_chunks:
                    yield c

            return gen()
        else:
            return next(static_iter)

    mock_completions.create.side_effect = mock_create

    harness = AgenticHarness(client=mock_openai)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "hard logic question"}
    ]

    result = await harness.generate_completion(messages=messages)

    assert mock_completions.create.call_count == 5
    assert result.model == AGENT_MODEL_NAME
    assert result.choices[0].message.content == "Consensus answer"


@pytest.mark.asyncio
async def test_agentic_harness_streaming_certain() -> None:
    """Verify streaming works properly when probe is certain."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    chunks = [
        make_mock_chunk("Stream certain", [(0.0, -2.0)]),
    ]

    async def mock_stream(*args: Any, **kwargs: Any) -> Any:
        async def gen() -> AsyncGenerator[Any, None]:
            for c in chunks:
                yield c

        return gen()

    mock_completions.create.side_effect = mock_stream

    harness = AgenticHarness(client=mock_openai)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "stream this"}
    ]

    chunks_received = []
    async for chunk in harness.generate_stream(messages=messages):
        chunks_received.append(chunk)

    assert mock_completions.create.call_count == 1
    assert len(chunks_received) > 0
    assert chunks_received[-1].choices[0].finish_reason == "stop"
    assert (
        "".join(c.choices[0].delta.content or "" for c in chunks_received)
        == "Stream certain"
    )
    assert all(c.model == AGENT_MODEL_NAME for c in chunks_received)


@pytest.mark.asyncio
async def test_agentic_harness_streaming_hesitant() -> None:
    """Verify streaming works properly when probe is hesitant."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    probe_chunks = [
        make_mock_chunk("Hesitant", [(0.0, -1.0)]),
    ]

    fork1_choice = MagicMock()
    fork1_choice.message.content = "A"
    fork1_resp = MagicMock()
    fork1_resp.choices = [fork1_choice]

    fork2_choice = MagicMock()
    fork2_choice.message.content = "B"
    fork2_resp = MagicMock()
    fork2_resp.choices = [fork2_choice]

    fork3_choice = MagicMock()
    fork3_choice.message.content = "C"
    fork3_resp = MagicMock()
    fork3_resp.choices = [fork3_choice]

    consensus_choice = MagicMock()
    consensus_choice.message.content = "Consensus stream answer"
    consensus_resp = MagicMock()
    consensus_resp.choices = [consensus_choice]
    consensus_resp.model = BACKEND_MODEL

    static_responses = [fork1_resp, fork2_resp, fork3_resp, consensus_resp]
    static_iter = iter(static_responses)

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):

            async def gen() -> AsyncGenerator[Any, None]:
                for c in probe_chunks:
                    yield c

            return gen()
        else:
            return next(static_iter)

    mock_completions.create.side_effect = mock_create

    harness = AgenticHarness(client=mock_openai)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "stream hesitant"}
    ]

    chunks_received = []
    async for chunk in harness.generate_stream(messages=messages):
        chunks_received.append(chunk)

    assert mock_completions.create.call_count == 5
    assert len(chunks_received) > 0
    assert chunks_received[-1].choices[0].finish_reason == "stop"
    assert (
        "".join(c.choices[0].delta.content or "" for c in chunks_received)
        == "Consensus stream answer"
    )
    assert all(c.model == AGENT_MODEL_NAME for c in chunks_received)


def test_gateway_health_endpoint() -> None:
    """Test the /health endpoint handles backend failures gracefully."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway_status"] == "ok"
    assert data["configured_agent"] == AGENT_MODEL_NAME


def test_gateway_models_endpoint() -> None:
    """Test the /v1/models endpoint reports the Triduum model."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    model_ids = [m["id"] for m in data["data"]]
    assert AGENT_MODEL_NAME in model_ids
