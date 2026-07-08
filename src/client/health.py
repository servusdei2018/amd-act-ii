import asyncio
import httpx
from openai import AsyncOpenAI
from src.config import VLLM_BASE_URL, GATEWAY_PORT, BACKEND_MODEL, AGENT_MODEL_NAME
from src.logging_config import logger

type HealthResults = dict[str, str | bool]


async def check_vllm_models(results: HealthResults, client: AsyncOpenAI) -> None:
    """Queries active models on the vLLM instance."""
    try:
        logger.info("Initiating query to check active models on local backend...")
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        results["vllm_models_status"] = (
            "ok" if BACKEND_MODEL in model_ids else "partial"
        )
        results["vllm_available_models"] = ", ".join(model_ids)
    except Exception as e:
        results["vllm_models_status"] = f"failed: {e}"


async def check_vllm_chat(results: HealthResults, client: AsyncOpenAI) -> None:
    """Verifies generative capabilities directly on the backend vLLM instance."""
    try:
        logger.info(
            f"Initiating chat completion test to direct backend model '{BACKEND_MODEL}'..."
        )
        response = await client.chat.completions.create(
            model=BACKEND_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        content = response.choices[0].message.content
        results["vllm_chat_status"] = "ok" if content else "empty_response"
        results["vllm_test_response"] = content or ""
    except Exception as e:
        results["vllm_chat_status"] = f"failed: {e}"


async def check_gateway_models(results: HealthResults, client: AsyncOpenAI) -> None:
    """Queries models list exposed by the gateway."""
    try:
        logger.info("Initiating query to check active models on gateway...")
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        results["gateway_models_status"] = (
            "ok" if AGENT_MODEL_NAME in model_ids else "failed"
        )
        results["gateway_available_models"] = ", ".join(model_ids)
    except Exception as e:
        results["gateway_models_status"] = f"failed: {e}"


async def check_gateway_chat(results: HealthResults, client: AsyncOpenAI) -> None:
    """Verifies chat completions through the gateway for Triduum."""
    try:
        logger.info(
            f"Initiating chat completion test to gateway agent model '{AGENT_MODEL_NAME}'..."
        )
        response = await client.chat.completions.create(
            model=AGENT_MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        content = response.choices[0].message.content
        results["gateway_chat_status"] = "ok" if content else "empty_response"
        results["gateway_test_response"] = content or ""
    except Exception as e:
        results["gateway_chat_status"] = f"failed: {e}"


async def run_health_checks() -> bool:
    """Orchestrates health verification against both backend and gateway endpoints."""
    logger.info("Executing API server connection and performance verification...")
    results: HealthResults = {}

    backend_client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="local-dummy-key")
    gateway_client = AsyncOpenAI(
        base_url=f"http://localhost:{GATEWAY_PORT}/v1", api_key="local-dummy-key"
    )

    logger.info("--- Stage 1: Verifying Backend vLLM Server ---")
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(check_vllm_models(results, backend_client))
            tg.create_task(check_vllm_chat(results, backend_client))
    except Exception as exc:
        logger.error(f"Backend verification TaskGroup execution failed: {exc}")
        return False

    backend_ok = False
    match (results.get("vllm_models_status"), results.get("vllm_chat_status")):
        case ("ok", "ok"):
            logger.info(
                "Direct backend connection succeeded: vLLM engine fully operational."
            )
            logger.info(
                f"Backend response payload: {results.get('vllm_test_response')}"
            )
            backend_ok = True
        case ("partial", "ok"):
            logger.warning(
                f"Warning: Direct backend chat succeeded, but primary target model '{BACKEND_MODEL}' "
                f"was not listed. Active backend models: {results.get('vllm_available_models')}"
            )
            backend_ok = True
        case (str() as model_error, _) if "failed" in model_error:
            logger.error(f"Backend model verification check failed: {model_error}")
        case (_, str() as chat_error) if "failed" in chat_error:
            logger.error(f"Backend chat execution check failed: {chat_error}")
        case _:
            logger.error(f"Backend execution finished in unexpected state: {results}")

    logger.info(f"--- Stage 2: Verifying Gateway & '{AGENT_MODEL_NAME}' Model ---")
    gateway_running = False
    try:
        async with httpx.AsyncClient() as http_client:
            res = await http_client.get(
                f"http://localhost:{GATEWAY_PORT}/health", timeout=1.0
            )
            if res.status_code == 200:
                gateway_running = True
    except Exception:
        logger.error(
            f"Gateway is not listening on port {GATEWAY_PORT}. Ensure the gateway is running."
        )
        return False

    if gateway_running:
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(check_gateway_models(results, gateway_client))
                tg.create_task(check_gateway_chat(results, gateway_client))
        except Exception as exc:
            logger.error(f"Gateway verification TaskGroup execution failed: {exc}")
            return False

        match (
            results.get("gateway_models_status"),
            results.get("gateway_chat_status"),
        ):
            case ("ok", "ok"):
                logger.info(
                    f"Gateway verification succeeded: '{AGENT_MODEL_NAME}' is operational."
                )
                logger.info(
                    f"Gateway response payload: {results.get('gateway_test_response')}"
                )
                return backend_ok
            case (str() as m_err, _) if "failed" in m_err:
                logger.error(f"Gateway model check failed: {m_err}")
                return False
            case (_, str() as c_err) if "failed" in c_err:
                logger.error(f"Gateway chat completion failed: {c_err}")
                return False
            case _:
                logger.error(
                    f"Gateway execution finished in unexpected state: {results}"
                )
                return False
    return False
