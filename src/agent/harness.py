import asyncio
import time
from copy import deepcopy
from typing import AsyncGenerator, Any, cast
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import (
    Choice as ChatCompletionChoice,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from src.config import (
    BACKEND_MODEL,
    AGENT_MODEL_NAME,
    VLLM_BASE_URL,
    PROBE_CERTAINTY_THRESHOLD,
    COT_FORK_TEMPERATURES,
    COT_SYSTEM_PROMPT,
    COT_FORK_PROMPTS,
    CONSENSUS_SYSTEM_PROMPT,
)
from src.logging_config import logger


class AgenticHarness:
    client: AsyncOpenAI

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self.client = client or AsyncOpenAI(
            base_url=VLLM_BASE_URL, api_key="local-dummy-key"
        )

    async def _run_probe(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> tuple[Any, bool, float]:
        """Executes a streaming completion check with logprobs enabled to determine certainty."""
        logger.info(f"[{AGENT_MODEL_NAME.upper()} - Probe] Starting Lean Probe Task...")

        probe_kwargs = kwargs.copy()
        probe_kwargs["model"] = BACKEND_MODEL
        probe_kwargs["stream"] = True
        probe_kwargs["logprobs"] = True
        probe_kwargs["top_logprobs"] = 2

        response_stream = await self.client.chat.completions.create(
            messages=messages, **probe_kwargs
        )

        accumulated_content = []
        deltas = []
        is_certain = True

        async for chunk in response_stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]

            if choice.delta and choice.delta.content:
                accumulated_content.append(choice.delta.content)

            if choice.logprobs and choice.logprobs.content:
                for token_logprob in choice.logprobs.content:
                    if (
                        token_logprob.top_logprobs
                        and len(token_logprob.top_logprobs) >= 2
                    ):
                        p1 = token_logprob.top_logprobs[0].logprob
                        p2 = token_logprob.top_logprobs[1].logprob
                        if p1 is not None and p2 is not None:
                            margin = p1 - p2
                            deltas.append(margin)

            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                if avg_delta <= PROBE_CERTAINTY_THRESHOLD:
                    is_certain = False
                    logger.info(
                        f"[{AGENT_MODEL_NAME.upper()} - Probe] Average Top-1 vs Top-2 Logprob Margin: {avg_delta:.4f} "
                        f"<= {PROBE_CERTAINTY_THRESHOLD:.4f}. Low confidence detected. Instantly halting probe stream after {len(deltas)} tokens."
                    )
                    break

        avg_logprob = sum(deltas) / len(deltas) if deltas else 0.0

        if not deltas:
            is_certain = False
            logger.warning(
                f"[{AGENT_MODEL_NAME.upper()} - Probe] No logprobs returned from backend. Defaulting to uncertain."
            )

        probe_response = None
        if is_certain:
            content_str = "".join(accumulated_content)
            logger.info(
                f"[{AGENT_MODEL_NAME.upper()} - Probe] Probe completed cleanly with high confidence. "
                f"Average Logprob Margin: {avg_logprob:.4f} (threshold: {PROBE_CERTAINTY_THRESHOLD:.4f})."
            )
            probe_response = ChatCompletion(
                id="chatcmpl-probe-certain",
                choices=[
                    ChatCompletionChoice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            content=content_str, role="assistant"
                        ),
                        logprobs=None,
                    )
                ],
                created=int(time.time()),
                model=BACKEND_MODEL,
                object="chat.completion",
            )

        return probe_response, is_certain, avg_logprob

    async def _run_cot_fork(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Runs a single CoT reasoning fork with a specific temperature."""
        logger.info(
            f"[{AGENT_MODEL_NAME.upper()} - Parallel Fork] Spawning CoT fork at temp {temperature}..."
        )

        # Clone messages to keep primary state untouched
        fork_messages = deepcopy(messages)

        # Find if there is an existing system message
        system_idx = -1
        for i, msg in enumerate(fork_messages):
            if msg.get("role") == "system":
                system_idx = i
                break

        cot_prompt = COT_FORK_PROMPTS.get(temperature, COT_SYSTEM_PROMPT)

        if system_idx >= 0:
            original_content = fork_messages[system_idx].get("content") or ""
            new_sys_msg = dict(fork_messages[system_idx])
            new_sys_msg["content"] = f"{original_content}\n\n{cot_prompt}"
            fork_messages[system_idx] = cast(ChatCompletionMessageParam, new_sys_msg)
        else:
            fork_messages.insert(0, {"role": "system", "content": cot_prompt})

        fork_kwargs = kwargs.copy()
        fork_kwargs["model"] = BACKEND_MODEL
        fork_kwargs["temperature"] = temperature
        fork_kwargs["stream"] = False

        response = await self.client.chat.completions.create(
            messages=fork_messages, **fork_kwargs
        )
        content = response.choices[0].message.content or ""
        logger.info(
            f"[{AGENT_MODEL_NAME.upper()} - Parallel Fork] Fork at temp {temperature} completed. "
            f"Response length: {len(content)} chars."
        )
        return content

    async def _run_consensus(
        self,
        original_messages: list[ChatCompletionMessageParam],
        fork_responses: list[str],
        **kwargs: Any,
    ) -> Any:
        """Invokes the Consensus Agent to evaluate and synthesize a final verified answer."""
        logger.info(
            f"[{AGENT_MODEL_NAME.upper()} - Consensus] Synthesizing consensus from forks..."
        )

        history_str = ""
        for msg in original_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_str += f"[{role.upper()}]: {content}\n"

        candidates_str = ""
        for i, resp in enumerate(fork_responses):
            candidates_str += f"--- Candidate {chr(65 + i)} ---\n{resp}\n\n"

        user_content = (
            f"Original Conversation History:\n{history_str}\n"
            f"Here are the three generated candidate reasoning paths and answers:\n\n{candidates_str}"
            f"Please evaluate them, resolve any errors or conflicts, and output only the final verified answer."
        )

        consensus_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": CONSENSUS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        consensus_kwargs = kwargs.copy()
        consensus_kwargs["model"] = BACKEND_MODEL
        consensus_kwargs["stream"] = False
        consensus_kwargs["temperature"] = 0.0  # Deterministic synthesis

        response = await self.client.chat.completions.create(
            messages=consensus_messages, **consensus_kwargs
        )
        logger.info(
            f"[{AGENT_MODEL_NAME.upper()} - Consensus] Consensus synthesis complete."
        )
        return response

    async def generate_completion(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> Any:
        """Non-streaming completions wrapper executing agentic scaffold tasks."""
        logger.info(
            f"[{AGENT_MODEL_NAME.upper()}] Invoking agentic harness with {len(messages)} messages."
        )

        probe_response, is_certain, _ = await self._run_probe(messages, **kwargs)
        if is_certain:
            logger.info(
                f"[{AGENT_MODEL_NAME.upper()}] Probe task was certain. Exiting instantly with answer."
            )
            probe_response.model = AGENT_MODEL_NAME
            return probe_response

        # Parallel CoTs
        logger.info(
            f"[{AGENT_MODEL_NAME.upper()}] Probe was hesitant. Triggering parallel CoT surge..."
        )

        tasks = [
            self._run_cot_fork(messages, temp, **kwargs)
            for temp in COT_FORK_TEMPERATURES
        ]
        fork_responses = await asyncio.gather(*tasks)

        # Consensus Agent
        consensus_response = await self._run_consensus(
            messages, fork_responses, **kwargs
        )
        consensus_response.model = AGENT_MODEL_NAME

        logger.info(
            f"[{AGENT_MODEL_NAME.upper()}] Agentic harness request successfully resolved via consensus."
        )
        return consensus_response

    async def generate_stream(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Streaming completions wrapper executing agentic scaffold tasks."""
        logger.info(
            f"[{AGENT_MODEL_NAME.upper()}] Invoking agentic harness in streaming mode with {len(messages)} messages."
        )

        probe_response, is_certain, _ = await self._run_probe(messages, **kwargs)
        if is_certain:
            logger.info(
                f"[{AGENT_MODEL_NAME.upper()}] Probe task was certain. Streaming cached probe answer."
            )
            content_str = probe_response.choices[0].message.content or ""
        else:
            logger.info(
                f"[{AGENT_MODEL_NAME.upper()}] Probe was hesitant. Resolving consensus before streaming."
            )
            tasks = [
                self._run_cot_fork(messages, temp, **kwargs)
                for temp in COT_FORK_TEMPERATURES
            ]
            fork_responses = await asyncio.gather(*tasks)

            consensus_response = await self._run_consensus(
                messages, fork_responses, **kwargs
            )
            content_str = consensus_response.choices[0].message.content or ""

        # Yield the entire content in a single chunk, followed by a stop chunk
        yield ChatCompletionChunk(
            id="chatcmpl-agentic-stream",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=content_str, role="assistant"),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=int(time.time()),
            model=AGENT_MODEL_NAME,
            object="chat.completion.chunk",
        )

        yield ChatCompletionChunk(
            id="chatcmpl-agentic-stream",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="", role="assistant"),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=int(time.time()),
            model=AGENT_MODEL_NAME,
            object="chat.completion.chunk",
        )

        logger.info(
            f"[{AGENT_MODEL_NAME.upper()}] Agentic stream successfully completed."
        )
