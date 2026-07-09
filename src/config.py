import os

# Backend server vLLM config
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
BACKEND_MODEL: str = os.getenv("BACKEND_MODEL", "cyankiwi/gemma-4-31B-it-AWQ-4bit")

# Gateway server config
GATEWAY_HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "8001"))
AGENT_MODEL_NAME: str = os.getenv("AGENT_MODEL_NAME", "triduum")

# Harness Agentic Scaffold Config
PROBE_CERTAINTY_THRESHOLD: float = float(os.getenv("PROBE_CERTAINTY_THRESHOLD", "1.5"))
COT_FORK_TEMPERATURES: list[float] = [0.1, 0.5, 0.8]
COT_SYSTEM_PROMPT: str = os.getenv(
    "COT_SYSTEM_PROMPT",
    "You are a reasoning agent fork. Please provide a detailed step-by-step reasoning chain (Chain of Thought) before outputting the final answer. Keep your thoughts clear and structured.",
)
COT_FORK_PROMPTS: dict[float, str] = {
    0.1: (
        "Deconstruct the input down to its absolute, primitive variables and explicit constraints. "
        "Trace your steps sequentially, validating each deduction against the source text before proceeding. "
        "Provide a precise, logical conclusion."
    ),
    0.5: (
        "Break this problem into discrete sub-problems. For each sub-problem, state your core assumptions clearly. "
        "Explore potential edge cases or alternative meanings in the instructions, and reconcile them before formatting your final answer."
    ),
    0.8: (
        "Approach this problem by identifying where a typical model would fail or misinterpret the instructions. "
        "Play devil's advocate against the most obvious solution, test an unusual perspective, and carefully build up to a robust conclusion."
    ),
}

CONSENSUS_AXES: list[str] = ["auditor", "tracker", "compiler"]

CONSENSUS_AXIS_PROMPTS: dict[str, str] = {
    "auditor": os.getenv(
        "CONSENSUS_AXIS_AUDITOR_PROMPT",
        (
            "You are Axis 1: The Logical Constraint Auditor. Your sole job is to cross-examine "
            "the three provided pathways (A, B, C) against the original user task. "
            "Identify every explicit rule, negative constraint (e.g., 'remove', 'omit', 'do not'), "
            "and logical boundary. Point out exactly which pathways failed to obey these constraints. "
            "Output a clear list of disqualified pathways and explain why they failed."
        ),
    ),
    "tracker": os.getenv(
        "CONSENSUS_AXIS_TRACKER_PROMPT",
        (
            "You are Axis 2: The Sequential State Tracker. Your sole job is to audit physical properties, "
            "mathematical updates, timeline sequences, or variable mutations across the three pathways (A, B, C). "
            "Trace the lifecycle of every object, entity, or variable from start to finish. Identify where a thread "
            "hallucinated a state, failed basic math, or skipped a step. Output the verified chronological "
            "truth table or sequence of states for the variables/entities. If the task does not involve quantitative "
            "or state mutations, trace the factual consistency, logic flow, and key assertions across the pathways instead."
        ),
    ),
    "compiler": os.getenv(
        "CONSENSUS_AXIS_COMPILER_PROMPT",
        (
            "You are Axis 3: The Syntax & Format Compiler. Your sole job is to enforce structural correctness "
            "based on the successful reasoning elements of the pathways. If the user prompt required JSON, "
            "Python lists, text code, or markdown tables, draft the final schema. Ensure all keys, types, "
            "and structures align perfectly with the raw data requirements. Output ONLY the compiled schema blueprint. "
            "If the task is open-ended or does not specify a format, outline the optimal layout, structure, and formatting "
            "standards to be used in the final response."
        ),
    ),
}

SYNTHESIS_SYSTEM_PROMPT: str = os.getenv(
    "SYNTHESIS_SYSTEM_PROMPT",
    (
        "You are the Core Synthesis Engine. You are given the original user task, the three raw CoT pathways, "
        "and the independent evaluations from the three specialized Consensus Axes (Auditor, Tracker, and Compiler). "
        "Your objective is to combine these vectors. Use the Auditor's disqualifications to prune the bad paths, "
        "the Tracker's table or factual trace to ensure accuracy, and the Compiler's layout to generate the output. "
        "Produce the final, pristine response. Do not include any meta-commentary, introductory remarks, or markdown code blocks."
    ),
)
