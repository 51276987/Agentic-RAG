"""Asynchronous API client for OpenViking knowledge base operations.

The client is intended for dependency injection into LangGraph nodes. It is
not registered as an LLM tool, so graph routing, authorization, retries, and
retrieval limits remain deterministic.
"""

from typing import (
    Any,
    Literal,
)
from urllib.parse import urlsplit

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


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


class OpenVikingKnowledgeAPI:
    """Type-safe asynchronous API for OpenViking knowledge resources."""

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
        if not settings.OPENVIKING_ENABLED:
            raise RuntimeError("OpenViking API 未启用，请设置 OPENVIKING_ENABLED=true")
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

    async def find(
        self,
        query: str,
        target_uri: str = "viking://resources",
        limit: int = 8,
    ) -> Any:
        """Perform deterministic semantic retrieval over knowledge resources."""
        normalized_query = query.strip()
        if not normalized_query or len(normalized_query) > 2000:
            raise ValueError("query 长度必须在 1 到 2000 个字符之间")
        return await self._request(
            "POST",
            "/api/v1/search/find",
            json_body={
                "query": normalized_query,
                "target_uri": _validate_resource_uri(target_uri),
                "context_type": ["resource"],
                "limit": _bounded_int(limit, name="limit", minimum=1, maximum=20),
            },
        )

    async def list_resources(
        self,
        uri: str = "viking://resources",
        recursive: bool = False,
        node_limit: int = 100,
    ) -> Any:
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

    async def read(
        self,
        uri: str,
        level: Literal["abstract", "overview", "full"] = "full",
        offset: int = 0,
        limit: int = 200,
    ) -> Any:
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
        to_uri: str | None = None,
        reason: str | None = None,
        wait: bool = False,
        timeout_seconds: float = 180,
        crawl_depth: int = 0,
        max_pages: int = 10,
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
        mode: Literal["create", "replace", "append"] = "create",
        wait: bool = False,
        timeout_seconds: float = 180,
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

    async def delete(
        self,
        uri: str,
        *,
        confirmed: bool = False,
        recursive: bool = False,
    ) -> Any:
        """Delete a knowledge resource after caller-side confirmation."""
        if not confirmed:
            raise ValueError("删除知识资源前必须由调用方完成权限校验和 HITL 确认")
        return await self._request(
            "DELETE",
            "/api/v1/fs",
            params={
                "uri": _validate_resource_uri(uri, allow_root=False),
                "recursive": recursive,
            },
        )


openviking_knowledge_api = OpenVikingKnowledgeAPI()
