"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse

from app.api.v1.auth import get_current_session
from app.core.config import settings
from app.core.langgraph.graph import LangGraphAgent
from app.core.limiter import limiter
from app.core.logging import logger
from app.core.metrics import llm_stream_duration_seconds
from app.models.session import Session
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    StreamDirectory,
    StreamOption,
    StreamResponse,
)
from app.services.session_naming import maybe_name_session

router = APIRouter()
agent = LangGraphAgent()


def _stream_response_from_chunk(chunk: str) -> StreamResponse:
    """Promote structured HITL JSON from content to frontend-facing fields."""
    try:
        payload = json.loads(chunk)
    except json.JSONDecodeError:
        return StreamResponse(event="message", content=chunk, done=False)
    if not isinstance(payload, dict):
        return StreamResponse(event="message", content=chunk, done=False)

    if payload.get("type") == "tool_progress":
        tool_name = payload.get("tool_name")
        tool_kind = payload.get("tool_kind")
        tool_status = payload.get("tool_status")
        title = payload.get("title")
        if (
            not isinstance(tool_name, str)
            or tool_kind not in {"node", "openviking"}
            or tool_status not in {"started", "completed", "failed"}
            or not isinstance(title, str)
        ):
            return StreamResponse(event="message", content=chunk, done=False)
        metadata = payload.get("metadata")
        return StreamResponse(
            event="tool",
            tool_name=tool_name,
            tool_kind=tool_kind,
            tool_status=tool_status,
            title=title,
            metadata=metadata if isinstance(metadata, dict) else None,
            done=False,
        )

    hitl_type = payload.get("type")
    title = str(payload.get("question") or "需要您补充信息")
    if hitl_type == "question_clarification":
        directories = [
            StreamDirectory(title=str(option["label"]), uri=str(option["value"]))
            for option in payload.get("system_options", [])
            if isinstance(option, dict) and option.get("label") and option.get("value")
        ]
        return StreamResponse(
            event="hitl",
            hitl_type="question_clarification",
            title=title,
            directories=directories,
            done=False,
        )
    if hitl_type == "role_clarification":
        options = [
            StreamOption(title=str(option["label"]), value=str(option["value"]))
            for option in payload.get("options", [])
            if isinstance(option, dict) and option.get("label") and option.get("value")
        ]
        return StreamResponse(
            event="hitl",
            hitl_type="role_clarification",
            title=title,
            options=options,
            done=False,
        )
    return StreamResponse(event="message", content=chunk, done=False)


def _format_stream_event(response: StreamResponse) -> str:
    """Serialize frontend event types with their stable SSE prefixes."""
    prefixes = {
        "message": "data",
        "tool": "tool",
        "hitl": "hitl",
        "error": "error",
        "done": "data",
    }
    prefix = prefixes[response.event]
    payload = json.dumps(
        response.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
    )
    return f"{prefix}: {payload}\n\n"


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        ChatResponse: The processed chat response.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        if settings.SESSION_NAMING_ENABLED:
            maybe_name_session(session.id, session.name, chat_request.messages)

        result = await agent.get_response(
            chat_request.messages, session.id, user_id=str(session.user_id), username=session.username
        )

        logger.info("chat_request_processed", session_id=session.id)

        return ChatResponse(messages=result)
    except Exception as e:
        logger.exception("chat_request_failed", session_id=session.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph with streaming response.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        StreamingResponse: A streaming response of the chat completion.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    logger.info(
        "stream_chat_request_received",
        session_id=session.id,
        message_count=len(chat_request.messages),
    )

    async def event_generator():
        """Generate message, HITL, error, and completion SSE events."""
        try:
            if settings.SESSION_NAMING_ENABLED:
                maybe_name_session(session.id, session.name, chat_request.messages)

            with llm_stream_duration_seconds.labels(model=agent.llm_service.get_llm().get_name()).time():
                async for chunk in agent.get_stream_response(
                    chat_request.messages, session.id, user_id=str(session.user_id), username=session.username
                ):
                    response = _stream_response_from_chunk(chunk)
                    yield _format_stream_event(response)

            # Send final message indicating successful completion.
            final_response = StreamResponse(event="done", content="", done=True)
            yield _format_stream_event(final_response)

        except Exception as e:
            logger.exception(
                "stream_chat_request_failed",
                session_id=session.id,
                error=str(e),
            )
            # Keep implementation details in logs while giving clients a
            # stable, safe terminal SSE payload.
            error_response = StreamResponse(
                event="error",
                content="服务内部错误，请稍后重试。",
                done=True,
            )
            yield _format_stream_event(error_response)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Get all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        ChatResponse: All messages in the session.

    Raises:
        HTTPException: If there's an error retrieving the messages.
    """
    try:
        messages = await agent.get_chat_history(session.id)
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.exception("get_messages_failed", session_id=session.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Clear all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        dict: A message indicating the chat history was cleared.
    """
    try:
        await agent.clear_chat_history(session.id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.exception("clear_chat_history_failed", session_id=session.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
