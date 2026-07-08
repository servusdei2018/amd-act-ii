from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from cli import (
    generate_chat_response,
    main,
    print_welcome_banner,
    supports_color,
)


def test_supports_color() -> None:
    """Test supports_color returns expected boolean based on TTY and environment."""
    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("os.getenv", return_value="xterm-256color"),
    ):
        assert supports_color() is True

    with (
        patch("sys.stdout.isatty", return_value=False),
        patch("os.getenv", return_value="xterm-256color"),
    ):
        assert supports_color() is False

    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("os.getenv", return_value="dumb"),
    ):
        assert supports_color() is False


@pytest.mark.asyncio
async def test_generate_chat_response_stream() -> None:
    """Verify generate_chat_response streams chunks and returns full content."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    chunk1 = ChatCompletionChunk(
        id="chunk-1",
        choices=[
            Choice(
                delta=ChoiceDelta(content="Hello", role="assistant"),
                finish_reason=None,
                index=0,
            )
        ],
        created=123456,
        model="triduum",
        object="chat.completion.chunk",
    )
    chunk2 = ChatCompletionChunk(
        id="chunk-2",
        choices=[
            Choice(
                delta=ChoiceDelta(content=" world!", role="assistant"),
                finish_reason=None,
                index=0,
            )
        ],
        created=123457,
        model="triduum",
        object="chat.completion.chunk",
    )

    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncMock:
        mock_iterator = MagicMock()
        mock_iterator.__aiter__.return_value = [chunk1, chunk2]
        return mock_iterator

    mock_completions.create.side_effect = mock_stream

    with (
        patch("cli.AsyncOpenAI", return_value=mock_openai),
        patch("sys.stdout.write") as mock_write,
        patch("sys.stdout.flush") as mock_flush,
    ):
        result = await generate_chat_response(
            base_url="http://localhost:8001/v1",
            model="triduum",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            stream=True,
        )

        assert result == "Hello world!"
        assert mock_write.call_count == 2
        mock_write.assert_any_call("Hello")
        mock_write.assert_any_call(" world!")
        assert mock_flush.call_count == 2


@pytest.mark.asyncio
async def test_generate_chat_response_non_stream() -> None:
    """Verify generate_chat_response requests non-streaming correctly."""
    mock_openai = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat.completions = mock_completions

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Static reply"
    mock_response.choices = [mock_choice]
    mock_completions.create.return_value = mock_response

    with (
        patch("cli.AsyncOpenAI", return_value=mock_openai),
        patch("sys.stdout.write") as mock_write,
    ):
        result = await generate_chat_response(
            base_url="http://localhost:8001/v1",
            model="triduum",
            messages=[{"role": "user", "content": "hi"}],
            temperature=None,
            stream=False,
        )

        assert result == "Static reply"
        mock_write.assert_called_once_with("Static reply")


def test_welcome_banner() -> None:
    """Verify printing of welcome banner does not raise errors."""
    with patch("sys.stdout.write"):
        print_welcome_banner("http://localhost:8001/v1", "triduum", True)


def test_cli_main_exit() -> None:
    """Verify main loop exits immediately on EOFError or exit command."""
    with (
        patch("sys.argv", ["cli.py"]),
        patch("builtins.input", side_effect=EOFError),
        patch("builtins.print") as mock_print,
    ):
        main()
        # Verify it printed exit message
        any_exiting = any(
            "Exiting chat client" in args[0]
            for args, _ in mock_print.call_args_list
            if args
        )
        assert any_exiting

    with (
        patch("sys.argv", ["cli.py"]),
        patch("builtins.input", side_effect=["/exit"]),
        patch("builtins.print") as mock_print,
    ):
        main()
        any_exiting = any(
            "Exiting chat client" in args[0]
            for args, _ in mock_print.call_args_list
            if args
        )
        assert any_exiting


def test_cli_main_commands() -> None:
    """Verify various slash commands in the REPL loop."""
    inputs = [
        "/info",
        "/model",
        "/model dummy-model",
        "/model",
        "/stream",
        "/stream off",
        "/stream on",
        "/system",
        "/system Custom system prompt",
        "/system",
        "/clear",
        "/reset",
        "/unknown-cmd",
        "/exit",
    ]

    with (
        patch("sys.argv", ["cli.py", "--system", "Initial prompt"]),
        patch("builtins.input", side_effect=inputs),
        patch("builtins.print") as mock_print,
    ):
        main()

        # Gather printed messages for assertion
        prints = [
            args[0]
            for args, _ in mock_print.call_args_list
            if args and isinstance(args[0], str)
        ]

        assert any("Model switched to: dummy-model" in p for p in prints)
        assert any("Streaming mode disabled." in p for p in prints)
        assert any("Streaming mode enabled." in p for p in prints)
        assert any(
            "System prompt updated. History cleared and reset." in p for p in prints
        )
        assert any("Conversation history cleared." in p for p in prints)
        assert any("Unknown command: /unknown-cmd" in p for p in prints)
