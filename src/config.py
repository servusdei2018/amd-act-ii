import os

# Backend server vLLM config
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
BACKEND_MODEL: str = os.getenv("BACKEND_MODEL", "cyankiwi/gemma-4-31B-it-AWQ-4bit")

# Gateway server config
GATEWAY_HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "8001"))
AGENT_MODEL_NAME: str = os.getenv("AGENT_MODEL_NAME", "triduum")

# Harness Agentic Scaffold Config
PROBE_CERTAINTY_THRESHOLD: float = float(os.getenv("PROBE_CERTAINTY_THRESHOLD", "-0.2"))
COT_FORK_TEMPERATURES: list[float] = [0.1, 0.5, 0.8]
COT_SYSTEM_PROMPT: str = os.getenv(
    "COT_SYSTEM_PROMPT",
    "You are a reasoning agent fork. Please provide a detailed step-by-step reasoning chain (Chain of Thought) before outputting the final answer. Keep your thoughts clear and structured.",
)
CONSENSUS_SYSTEM_PROMPT: str = os.getenv(
    "CONSENSUS_SYSTEM_PROMPT",
    "You are the Consensus Agent. You are given a user request and three candidate solutions generated with different temperatures/parameters. Some solutions might contain logical errors or inconsistencies.\n\nYour task is to:\n1. Carefully analyze each of the three reasoning paths and their final answers.\n2. Resolve any logic conflicts or errors.\n3. Synthesize the final, verified, pristine answer.\n4. Output ONLY the final verified answer, without repeating the candidate paths or adding meta-commentary.",
)
