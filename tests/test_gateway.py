import pytest
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


@pytest.mark.asyncio
async def test_agentic_harness_certain() -> None:
    """Verify that when the probe is certain, we exit instantly with the probe's answer."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    mock_token_logprob = MagicMock()
    mock_token_logprob.logprob = 0.0  # Above threshold

    mock_choice = MagicMock()
    mock_choice.logprobs = MagicMock()
    mock_choice.logprobs.content = [mock_token_logprob]
    mock_choice.message.content = "Certain answer"

    mock_response = MagicMock()
    mock_response.model = BACKEND_MODEL
    mock_response.choices = [mock_choice]

    mock_completions.create.return_value = mock_response

    harness = AgenticHarness(client=mock_openai)
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]

    result = await harness.generate_completion(messages=messages)

    assert mock_completions.create.call_count == 1
    kwargs = mock_completions.create.call_args.kwargs
    assert kwargs["logprobs"] is True
    assert result.model == AGENT_MODEL_NAME
    assert result.choices[0].message.content == "Certain answer"


@pytest.mark.asyncio
async def test_agentic_harness_hesitant() -> None:
    """Verify that when the probe is hesitant, we run parallel forks and consensus resolver."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    mock_token_logprob = MagicMock()
    mock_token_logprob.logprob = -1.5  # Below threshold

    probe_choice = MagicMock()
    probe_choice.logprobs = MagicMock()
    probe_choice.logprobs.content = [mock_token_logprob]
    probe_choice.message.content = "Hesitant draft"

    probe_resp = MagicMock()
    probe_resp.choices = [probe_choice]
    probe_resp.model = BACKEND_MODEL

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

    mock_completions.create.side_effect = [
        probe_resp,
        fork1_resp,
        fork2_resp,
        fork3_resp,
        consensus_resp,
    ]

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

    mock_token_logprob = MagicMock()
    mock_token_logprob.logprob = 0.0

    mock_choice = MagicMock()
    mock_choice.logprobs = MagicMock()
    mock_choice.logprobs.content = [mock_token_logprob]
    mock_choice.message.content = "Stream certain"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = BACKEND_MODEL

    mock_completions.create.return_value = mock_response

    harness = AgenticHarness(client=mock_openai)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "stream this"}
    ]

    chunks = []
    async for chunk in harness.generate_stream(messages=messages):
        chunks.append(chunk)

    assert mock_completions.create.call_count == 1
    assert len(chunks) > 0
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Stream certain"
    assert all(c.model == AGENT_MODEL_NAME for c in chunks)


@pytest.mark.asyncio
async def test_agentic_harness_streaming_hesitant() -> None:
    """Verify streaming works properly when probe is hesitant."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    mock_token_logprob = MagicMock()
    mock_token_logprob.logprob = -1.5

    probe_choice = MagicMock()
    probe_choice.logprobs = MagicMock()
    probe_choice.logprobs.content = [mock_token_logprob]
    probe_choice.message.content = "Hesitant draft"
    probe_resp = MagicMock()
    probe_resp.choices = [probe_choice]
    probe_resp.model = BACKEND_MODEL

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

    mock_completions.create.side_effect = [
        probe_resp,
        fork1_resp,
        fork2_resp,
        fork3_resp,
        consensus_resp,
    ]

    harness = AgenticHarness(client=mock_openai)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "stream hesitant"}
    ]

    chunks = []
    async for chunk in harness.generate_stream(messages=messages):
        chunks.append(chunk)

    assert mock_completions.create.call_count == 5
    assert len(chunks) > 0
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert (
        "".join(c.choices[0].delta.content or "" for c in chunks)
        == "Consensus stream answer"
    )
    assert all(c.model == AGENT_MODEL_NAME for c in chunks)


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
