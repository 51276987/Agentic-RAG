"""Observability module for the application."""

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from app.core.config import settings
from app.core.logging import logger

_langfuse_client: Langfuse | None = None
_langfuse_callback_handler: CallbackHandler | None = None


def _has_langfuse_credentials() -> bool:
    """Return whether non-placeholder Langfuse credentials are configured."""
    public_key = settings.LANGFUSE_PUBLIC_KEY.strip()
    secret_key = settings.LANGFUSE_SECRET_KEY.strip()
    return bool(
        public_key
        and secret_key
        and not public_key.startswith("your-")
        and not secret_key.startswith("your-")
    )


def langfuse_init() -> bool:
    """Initialize the shared Langfuse client and LangChain callback."""
    global _langfuse_callback_handler, _langfuse_client

    if not settings.LANGFUSE_TRACING_ENABLED:
        logger.debug("langfuse_tracing_disabled")
        return False

    if not _has_langfuse_credentials():
        logger.warning("langfuse_credentials_missing")
        return False

    if _langfuse_client is not None and _langfuse_callback_handler is not None:
        return True

    _langfuse_client = Langfuse(
        tracing_enabled=settings.LANGFUSE_TRACING_ENABLED,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        environment=settings.ENVIRONMENT.value,
        debug=settings.DEBUG,
    )
    _langfuse_callback_handler = CallbackHandler(public_key=settings.LANGFUSE_PUBLIC_KEY)

    try:
        if _langfuse_client.auth_check():
            logger.info("langfuse_auth_success", host=settings.LANGFUSE_HOST)
        else:
            logger.warning("langfuse_auth_failure")
    except Exception:
        logger.exception("langfuse_auth_check_failed")
        return False

    return True


def get_langfuse_callback_handler() -> CallbackHandler | None:
    """Return the shared callback when tracing is enabled and configured."""
    if _langfuse_callback_handler is None:
        langfuse_init()
    return _langfuse_callback_handler


def langfuse_shutdown() -> None:
    """Flush pending observations and stop the Langfuse client."""
    if _langfuse_client is None:
        return

    try:
        _langfuse_client.shutdown()
        logger.info("langfuse_shutdown_complete")
    except Exception:
        logger.exception("langfuse_shutdown_failed")
