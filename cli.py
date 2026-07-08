#!/usr/bin/env python3
"""Terminal CLI client to interact with the local Triduum Gateway."""

import argparse
import asyncio
import os
import sys
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.config import AGENT_MODEL_NAME, GATEWAY_HOST, GATEWAY_PORT

DEFAULT_HOST = "localhost" if GATEWAY_HOST == "0.0.0.0" else GATEWAY_HOST
DEFAULT_PORT = GATEWAY_PORT
DEFAULT_MODEL = AGENT_MODEL_NAME


def supports_color() -> bool:
    """Checks if the terminal supports ANSI color formatting."""
    return sys.stdout.isatty() and os.getenv("TERM") != "dumb"


class Colors:
    """ANSI color definitions for terminal formatting."""

    CYAN = "\033[36m" if supports_color() else ""
    GREEN = "\033[1;32m" if supports_color() else ""
    BLUE = "\033[1;34m" if supports_color() else ""
    RED = "\033[31m" if supports_color() else ""
    YELLOW = "\033[1;33m" if supports_color() else ""
    BOLD = "\033[1m" if supports_color() else ""
    RESET = "\033[0m" if supports_color() else ""


async def generate_chat_response(
    base_url: str,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float | None,
    stream: bool,
) -> str:
    """Invokes chat completions API from the Gateway server, handling streaming output."""
    client = AsyncOpenAI(base_url=base_url, api_key="local-dummy-key")

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        if stream:
            response_stream = await client.chat.completions.create(**kwargs)
            full_content = []
            async for chunk in response_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        sys.stdout.write(delta.content)
                        sys.stdout.flush()
                        full_content.append(delta.content)
            return "".join(full_content)
        else:
            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            sys.stdout.write(content)
            sys.stdout.flush()
            return content
    except Exception as e:
        print(f"\n{Colors.RED}Error calling Gateway API: {e}{Colors.RESET}")
        raise


def print_welcome_banner(base_url: str, model: str, stream: bool) -> None:
    """Prints a styled startup banner in the terminal."""
    banner_width = 80
    print("=" * banner_width)
    print(
        f"{Colors.BOLD}{Colors.CYAN}TRIDUUM GATEWAY CLIENT{Colors.RESET}".center(
            banner_width + (14 if supports_color() else 0)
        )
    )
    print("=" * banner_width)
    print(f"Connected to: {Colors.BOLD}{base_url}{Colors.RESET}")
    print(f"Active Model: {Colors.BOLD}{model}{Colors.RESET}")
    print(f"Stream Mode:  {Colors.BOLD}{'ON' if stream else 'OFF'}{Colors.RESET}")
    print()
    print("Commands:")
    print("  /exit, /quit       Exit the chat application")
    print("  /clear, /reset     Clear current conversation history")
    print("  /info              Show current settings and history statistics")
    print("  /system <prompt>   Change the system prompt (resets history)")
    print("  /model <name>      Switch the active model")
    print("  /stream <on|off>   Turn streaming mode on or off")
    print("=" * banner_width)
    print()


def main() -> None:
    """Main CLI orchestrator running the synchronous interactive REPL."""
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat client for Triduum Gateway"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Gateway host to connect to (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Gateway port to connect to (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to send requests to (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional initial system prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature for inference (e.g. 0.7)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    model = args.model
    system_prompt = args.system
    temperature = args.temperature
    stream = not args.no_stream

    print_welcome_banner(base_url, model, stream)

    # Initialize messages list
    messages: list[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            # Read input from the user
            user_input = input(f"{Colors.BLUE}{Colors.BOLD}User > {Colors.RESET}")
        except KeyboardInterrupt, EOFError:
            print(f"\n{Colors.CYAN}Exiting chat client. Goodbye!{Colors.RESET}")
            break

        cleaned_input = user_input.strip()
        if not cleaned_input:
            continue

        # Handle slash commands
        if cleaned_input.startswith("/"):
            cmd_parts = cleaned_input.split(" ", 1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                print(f"{Colors.CYAN}Exiting chat client. Goodbye!{Colors.RESET}")
                break

            elif cmd in ("/clear", "/reset"):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                print(f"{Colors.CYAN}Conversation history cleared.{Colors.RESET}\n")
                continue

            elif cmd == "/info":
                sys_msgs = [m["content"] for m in messages if m.get("role") == "system"]
                curr_sys = sys_msgs[0] if sys_msgs else "(none)"
                print(f"\n{Colors.CYAN}--- Client Info & Statistics ---{Colors.RESET}")
                print(f"Base URL:      {base_url}")
                print(f"Active Model:  {model}")
                print(f"Stream Mode:   {'ON' if stream else 'OFF'}")
                print(
                    f"Temperature:   {temperature if temperature is not None else 'Default'}"
                )
                print(f"System Prompt: {curr_sys}")
                print(
                    f"History Size:  {len(messages)} messages (including system context)"
                )
                print()
                continue

            elif cmd == "/system":
                if arg:
                    system_prompt = arg
                    messages = [{"role": "system", "content": system_prompt}]
                    print(
                        f"{Colors.CYAN}System prompt updated. History cleared and reset.{Colors.RESET}\n"
                    )
                else:
                    sys_msgs = [
                        m["content"] for m in messages if m.get("role") == "system"
                    ]
                    curr_sys = sys_msgs[0] if sys_msgs else "(none)"
                    print(
                        f"{Colors.CYAN}Current System Prompt: {curr_sys}{Colors.RESET}\n"
                    )
                continue

            elif cmd == "/model":
                if arg:
                    model = arg
                    print(f"{Colors.CYAN}Model switched to: {model}{Colors.RESET}\n")
                else:
                    print(f"{Colors.CYAN}Current active model: {model}{Colors.RESET}\n")
                continue

            elif cmd == "/stream":
                if arg.lower() in ("on", "true", "yes", "1"):
                    stream = True
                    print(f"{Colors.CYAN}Streaming mode enabled.{Colors.RESET}\n")
                elif arg.lower() in ("off", "false", "no", "0"):
                    stream = False
                    print(f"{Colors.CYAN}Streaming mode disabled.{Colors.RESET}\n")
                else:
                    print(
                        f"{Colors.CYAN}Streaming mode is currently: {'ON' if stream else 'OFF'}{Colors.RESET}\n"
                    )
                continue

            else:
                print(
                    f"{Colors.RED}Unknown command: {cmd}. Type /info to see settings.{Colors.RESET}\n"
                )
                continue

        messages.append({"role": "user", "content": cleaned_input})
        print(
            f"{Colors.GREEN}{Colors.BOLD}Assistant > {Colors.RESET}", end="", flush=True
        )

        try:
            assistant_content = asyncio.run(
                generate_chat_response(
                    base_url=base_url,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=stream,
                )
            )
            # Append successful assistant response to history
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            print("\n")

        except KeyboardInterrupt:
            # Let the user interrupt long stream outputs without exiting the REPL
            print(f"\n{Colors.YELLOW}[Generation interrupted by user]{Colors.RESET}\n")
            # If we interrupted, remove the user message so they can re-try without polluting history
            if messages and messages[-1]["role"] == "user":
                messages.pop()

        except Exception:
            # Other errors have been printed inside generate_chat_response
            # Remove the last user message so history isn't left in an incomplete state
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            print()


if __name__ == "__main__":
    main()
