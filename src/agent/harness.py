import asyncio
import time
from copy import deepcopy
from typing import AsyncGenerator, Any, cast
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from src.config import (
    BACKEND_MODEL,
    AGENT_MODEL_NAME,
    VLLM_BASE_URL,
    PROBE_CERTAINTY_THRESHOLD,
    COT_FORK_TEMPERATURES,
    COT_SYSTEM_PROMPT,
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
        """Executes a single completion check with logprobs enabled to determine certainty."""
        logger.info(f"[{AGENT_MODEL_NAME.upper()} - Probe] Starting Lean Probe Task...")

        probe_kwargs = kwargs.copy()
        probe_kwargs["model"] = BACKEND_MODEL
        probe_kwargs["stream"] = False
        probe_kwargs["logprobs"] = True

        response = await self.client.chat.completions.create(
            messages=messages, **probe_kwargs
        )

        choice = response.choices[0]
        avg_logprob = 0.0
        is_certain = False

        if choice.logprobs and choice.logprobs.content:
            logprobs = [
                t.logprob for t in choice.logprobs.content if t.logprob is not None
            ]
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                if avg_logprob >= PROBE_CERTAINTY_THRESHOLD:
                    is_certain = True
                logger.info(
                    f"[{AGENT_MODEL_NAME.upper()} - Probe] Average logprob: {avg_logprob:.4f} "
                    f"(threshold: {PROBE_CERTAINTY_THRESHOLD:.4f}). Certainty: {is_certain}"
                )
            else:
                logger.warning(
                    f"[{AGENT_MODEL_NAME.upper()} - Probe] Logprobs list was empty."
                )
        else:
            logger.warning(
                f"[{AGENT_MODEL_NAME.upper()} - Probe] No logprobs returned from backend."
            )

        return response, is_certain, avg_logprob

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

        if system_idx >= 0:
            original_content = fork_messages[system_idx].get("content") or ""
            new_sys_msg = dict(fork_messages[system_idx])
            new_sys_msg["content"] = f"{original_content}\n\n{COT_SYSTEM_PROMPT}"
            fork_messages[system_idx] = cast(ChatCompletionMessageParam, new_sys_msg)
        else:
            fork_messages.insert(0, {"role": "system", "content": COT_SYSTEM_PROMPT})

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
                Choice(
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
                Choice(
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
