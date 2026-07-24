"""OpenViking knowledge base tools for LangGraph.

Only resource-oriented operations are exposed. OpenViking administration,
sessions, memories, skills, snapshots, and system operations are intentionally
outside this module's scope.
"""

import json
from collections.abc import Awaitable
from typing import (
    Any,
    Literal,
)
from urllib.parse import urlsplit

import httpx
from langchain_core.tools import tool
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger


class OpenVikingAPIError(RuntimeError):
    """Represent a structured error returned by the OpenViking HTTP API."""

    def __init__(self, status_code: int, detail: str):
        """Initialize an OpenViking API error."""
        self.status_code = status_code
        super().__init__(f"OpenViking API returned HTTP {status_code}: {detail}")


def _should_retry_openviking(exc: BaseException) -> bool:
    """Return whether an OpenViking request failure is transient."""
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, OpenVikingAPIError) and exc.status_code >= 500


def _validate_resource_uri(uri: str, *, allow_root: bool = True) -> str:
    """Validate that a URI points to an OpenViking knowledge resource scope."""
    value = uri.strip()
    parsed = urlsplit(value)
    if parsed.scheme != "viking" or parsed.query or parsed.fragment:
        raise ValueError("URI 必须是合法的 viking:// 知识库资源 URI")

    path_parts = [part for part in parsed.path.split("/") if part]
    is_public_resource = parsed.netloc == "resources"
    is_user_resource = parsed.netloc == "user" and "resources" in path_parts
    if not is_public_resource and not is_user_resource:
        raise ValueError("仅允许访问 viking://resources 或 viking://user/.../resources 知识库范围")

    is_root = (is_public_resource and not path_parts) or (
        is_user_resource and path_parts[-1] == "resources"
    )
    if is_root and not allow_root:
        raise ValueError("该操作不允许直接作用于知识库根目录")
    return value


def _validate_source_url(source_url: str) -> str:
    """Allow remote HTTP(S) knowledge sources while rejecting local file paths."""
    value = source_url.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url 必须是可访问的 HTTP 或 HTTPS 地址")
    return value


def _bounded_int(value: int, *, name: str, minimum: int, maximum: int) -> int:
    """Validate a bounded integer tool argument."""
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return value


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    """Remove unset query parameters or JSON fields."""
    return {key: value for key, value in values.items() if value is not None}


class OpenVikingKnowledgeClient:
    """Small asynchronous client for OpenViking knowledge base APIs."""

    def __init__(self) -> None:
        """Initialize the client from application settings."""
        self._base_url = settings.OPENVIKING_BASE_URL.rstrip("/")
        self._api_key = settings.OPENVIKING_API_KEY
        self._auth_mode = settings.OPENVIKING_AUTH_MODE.lower()
        self._account = settings.OPENVIKING_ACCOUNT
        self._user = settings.OPENVIKING_USER
        self._timeout = settings.OPENVIKING_TIMEOUT_SECONDS

    def _headers(self) -> dict[str, str]:
        """Build authentication headers without exposing credentials to tools."""
        if self._auth_mode not in {"api_key", "trusted", "dev"}:
            raise ValueError("OPENVIKING_AUTH_MODE 必须是 api_key、trusted 或 dev")

        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._auth_mode == "api_key" and not self._api_key:
            raise ValueError("OpenViking api_key 模式需要配置 OPENVIKING_API_KEY")

        if self._auth_mode == "trusted":
            if not self._account or not self._user:
                raise ValueError("OpenViking trusted 模式需要配置 OPENVIKING_ACCOUNT 和 OPENVIKING_USER")
            headers["X-OpenViking-Account"] = self._account
            headers["X-OpenViking-User"] = self._user

        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_should_retry_openviking),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        request_timeout: float | None = None,
    ) -> Any:
        """Call OpenViking and unwrap its unified response envelope."""
        if not self._base_url:
            raise ValueError("未配置 OPENVIKING_BASE_URL")

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers(),
            timeout=request_timeout or self._timeout,
        ) as client:
            response = await client.request(method, path, params=params, json=json_body)

        if response.is_error:
            detail = response.text[:2000] or response.reason_phrase
            raise OpenVikingAPIError(response.status_code, detail)

        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenVikingAPIError(502, "响应不是合法 JSON") from exc

        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise OpenVikingAPIError(502, "响应缺少 status=ok")
        return payload.get("result")

    async def find(self, query: str, target_uri: str, limit: int) -> Any:
        """Perform deterministic semantic retrieval over knowledge resources."""
        return await self._request(
            "POST",
            "/api/v1/search/find",
            json_body={
                "query": query,
                "target_uri": _validate_resource_uri(target_uri),
                "context_type": ["resource"],
                "limit": _bounded_int(limit, name="limit", minimum=1, maximum=20),
            },
        )

    async def list_resources(self, uri: str, recursive: bool, node_limit: int) -> Any:
        """List knowledge files and directories."""
        return await self._request(
            "GET",
            "/api/v1/fs/ls",
            params={
                "uri": _validate_resource_uri(uri),
                "recursive": recursive,
                "output": "agent",
                "node_limit": _bounded_int(node_limit, name="node_limit", minimum=1, maximum=200),
            },
        )

    async def read(self, uri: str, level: str, offset: int, limit: int) -> Any:
        """Read L0, L1, or L2 knowledge content."""
        resource_uri = _validate_resource_uri(uri)
        if level == "abstract":
            return await self._request("GET", "/api/v1/content/abstract", params={"uri": resource_uri})
        if level == "overview":
            return await self._request("GET", "/api/v1/content/overview", params={"uri": resource_uri})
        if level != "full":
            raise ValueError("level 必须是 abstract、overview 或 full")
        return await self._request(
            "GET",
            "/api/v1/content/read",
            params={
                "uri": resource_uri,
                "offset": _bounded_int(offset, name="offset", minimum=0, maximum=1_000_000),
                "limit": _bounded_int(limit, name="limit", minimum=1, maximum=500),
            },
        )

    async def add_url(
        self,
        source_url: str,
        to_uri: str | None,
        reason: str | None,
        wait: bool,
        timeout_seconds: float,
        crawl_depth: int,
        max_pages: int,
    ) -> Any:
        """Import a URL into the knowledge base."""
        timeout = max(1.0, min(timeout_seconds, 600.0))
        body = _without_none(
            {
                "path": _validate_source_url(source_url),
                "to": _validate_resource_uri(to_uri, allow_root=False) if to_uri else None,
                "reason": reason.strip() if reason else None,
                "wait": wait,
                "timeout": timeout if wait else None,
                "args": {
                    "depth": _bounded_int(crawl_depth, name="crawl_depth", minimum=0, maximum=3),
                    "max_pages": _bounded_int(max_pages, name="max_pages", minimum=1, maximum=100),
                },
            }
        )
        return await self._request(
            "POST",
            "/api/v1/resources",
            json_body=body,
            request_timeout=max(self._timeout, timeout + 5) if wait else None,
        )

    async def write(
        self,
        uri: str,
        content: str,
        mode: str,
        wait: bool,
        timeout_seconds: float,
    ) -> Any:
        """Create, replace, or append a textual knowledge resource."""
        if mode not in {"create", "replace", "append"}:
            raise ValueError("mode 必须是 create、replace 或 append")
        if not content or len(content) > 50_000:
            raise ValueError("content 长度必须在 1 到 50000 个字符之间")

        timeout = max(1.0, min(timeout_seconds, 600.0))
        return await self._request(
            "POST",
            "/api/v1/content/write",
            json_body=_without_none(
                {
                    "uri": _validate_resource_uri(uri, allow_root=False),
                    "content": content,
                    "mode": mode,
                    "wait": wait,
                    "timeout": timeout if wait else None,
                }
            ),
            request_timeout=max(self._timeout, timeout + 5) if wait else None,
        )

    async def stat(self, uri: str) -> Any:
        """Get processing and filesystem status for a knowledge resource."""
        return await self._request(
            "GET",
            "/api/v1/fs/stat",
            params={"uri": _validate_resource_uri(uri)},
        )

    async def task_status(self, task_id: str) -> Any:
        """Get a background resource-import task status."""
        value = task_id.strip()
        if not value or "/" in value or len(value) > 200:
            raise ValueError("task_id 格式无效")
        return await self._request("GET", f"/api/v1/tasks/{value}")

    async def delete(self, uri: str, recursive: bool) -> Any:
        """Delete a knowledge resource after caller-side confirmation."""
        return await self._request(
            "DELETE",
            "/api/v1/fs",
            params={
                "uri": _validate_resource_uri(uri, allow_root=False),
                "recursive": recursive,
            },
        )


openviking_knowledge_client = OpenVikingKnowledgeClient()


def _serialize_tool_result(operation: str, result: Any) -> str:
    """Serialize and bound a tool result before returning it to the LLM."""
    payload = {"ok": True, "operation": operation, "result": result}
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    max_chars = settings.OPENVIKING_TOOL_MAX_OUTPUT_CHARS
    if len(serialized) <= max_chars:
        return serialized
    return json.dumps(
        {
            "ok": True,
            "operation": operation,
            "truncated": True,
            "result_preview": serialized[:max_chars],
        },
        ensure_ascii=False,
    )


async def _execute_tool(operation: str, request: Awaitable[Any]) -> str:
    """Execute an OpenViking request and return a model-readable result."""
    try:
        return _serialize_tool_result(operation, await request)
    except Exception as exc:
        logger.warning(
            "openviking_tool_failed",
            operation=operation,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return json.dumps(
            {
                "ok": False,
                "operation": operation,
                "error": str(exc),
            },
            ensure_ascii=False,
        )


@tool
async def openviking_find(
    query: str,
    target_uri: str = "viking://resources",
    limit: int = 8,
) -> str:
    """在 OpenViking 知识库中进行语义检索，只搜索 resource，不搜索 memory 或 skill."""
    normalized_query = query.strip()
    if not normalized_query or len(normalized_query) > 2000:
        return json.dumps({"ok": False, "error": "query 长度必须在 1 到 2000 个字符之间"}, ensure_ascii=False)
    return await _execute_tool(
        "find",
        openviking_knowledge_client.find(normalized_query, target_uri, limit),
    )


@tool
async def openviking_list_resources(
    uri: str = "viking://resources",
    recursive: bool = False,
    node_limit: int = 100,
) -> str:
    """列出 OpenViking 知识库目录；需要浏览知识库结构或定位文件 URI 时使用."""
    return await _execute_tool(
        "list_resources",
        openviking_knowledge_client.list_resources(uri, recursive, node_limit),
    )


@tool
async def openviking_read_resource(
    uri: str,
    level: Literal["abstract", "overview", "full"] = "full",
    offset: int = 0,
    limit: int = 200,
) -> str:
    """读取知识内容：abstract=L0 摘要，overview=L1 目录概览，full=L2 文件正文."""
    return await _execute_tool(
        "read_resource",
        openviking_knowledge_client.read(uri, level, offset, limit),
    )


@tool
async def openviking_add_url_resource(
    source_url: str,
    to_uri: str = "",
    reason: str = "",
    wait: bool = False,
    timeout_seconds: float = 180,
    crawl_depth: int = 0,
    max_pages: int = 10,
) -> str:
    """把 HTTP(S) 文档、网页或 Git 仓库 URL 导入 OpenViking 知识库."""
    return await _execute_tool(
        "add_url_resource",
        openviking_knowledge_client.add_url(
            source_url=source_url,
            to_uri=to_uri or None,
            reason=reason or None,
            wait=wait,
            timeout_seconds=timeout_seconds,
            crawl_depth=crawl_depth,
            max_pages=max_pages,
        ),
    )


@tool
async def openviking_write_resource(
    uri: str,
    content: str,
    mode: Literal["create", "replace", "append"] = "create",
    wait: bool = False,
    timeout_seconds: float = 180,
) -> str:
    """创建或更新 OpenViking 文本知识；仅支持资源文件 URI，不用于 memory、skill 或 session."""
    return await _execute_tool(
        "write_resource",
        openviking_knowledge_client.write(uri, content, mode, wait, timeout_seconds),
    )


@tool
async def openviking_resource_status(uri: str) -> str:
    """查看 OpenViking 知识文件或目录的状态、大小、类型和锁状态."""
    return await _execute_tool("resource_status", openviking_knowledge_client.stat(uri))


@tool
async def openviking_import_task_status(task_id: str) -> str:
    """查询异步知识导入返回的 task_id 当前处理状态."""
    return await _execute_tool(
        "import_task_status",
        openviking_knowledge_client.task_status(task_id),
    )


@tool
async def openviking_delete_resource(
    uri: str,
    confirmed: bool = False,
    recursive: bool = False,
) -> str:
    """删除知识资源；调用前必须先用 ask_human 获得用户确认，并将 confirmed 设为 true."""
    if not confirmed:
        return json.dumps(
            {
                "ok": False,
                "operation": "delete_resource",
                "error": "删除前必须调用 ask_human 获得用户明确确认",
            },
            ensure_ascii=False,
        )
    return await _execute_tool(
        "delete_resource",
        openviking_knowledge_client.delete(uri, recursive),
    )


openviking_knowledge_tools = [
    openviking_find,
    openviking_list_resources,
    openviking_read_resource,
    openviking_add_url_resource,
    openviking_write_resource,
    openviking_resource_status,
    openviking_import_task_status,
    openviking_delete_resource,
]
