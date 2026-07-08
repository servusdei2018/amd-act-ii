import argparse
import asyncio
import sys
import uvicorn
from src.config import GATEWAY_HOST, GATEWAY_PORT
from src.logging_config import logger


async def run_health_checks_wrapper() -> None:
    """Wrapper that executes client health checks and exits appropriately."""
    from src.client.health import run_health_checks

    try:
        success = await run_health_checks()
        if not success:
            logger.error("System verification checks completed with failures.")
            sys.exit(1)
        logger.info("All connection and integration checks completed successfully.")
    except Exception as e:
        logger.critical(f"Critical failure during verification run: {e}")
        sys.exit(1)


def main() -> None:
    """CLI orchestrator entrypoint."""
    parser = argparse.ArgumentParser(
        description="Triduum Model Gateway and Inference Infrastructure CLI"
    )
    parser.add_argument(
        "--health",
        "--check",
        action="store_true",
        help="Execute connection test verification checks against both vLLM and Gateway endpoints.",
    )
    args = parser.parse_args()

    if args.health:
        with asyncio.Runner() as runner:
            runner.run(run_health_checks_wrapper())
    else:
        logger.info(
            f"Launching Triduum Gateway on http://{GATEWAY_HOST}:{GATEWAY_PORT}..."
        )
        try:
            uvicorn.run(
                "src.server.app:app",
                host=GATEWAY_HOST,
                port=GATEWAY_PORT,
                log_config=None,
                access_log=True,
            )
        except Exception as e:
            logger.critical(f"Gateway server failed to start: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
