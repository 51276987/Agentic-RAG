# FastAPI LangGraph Agent 接口文档

本文档依据当前仓库源码编写。默认服务地址为：

```text
http://localhost:8000
```

默认 API 前缀为：

```text
/api/v1
```

交互式文档：

- Swagger UI：`http://localhost:8000/docs`
- ReDoc：`http://localhost:8000/redoc`
- OpenAPI JSON：`http://localhost:8000/api/v1/openapi.json`

## 1. 鉴权说明

系统使用 JWT Bearer Token，并区分两类 Token：

| Token 类型 | JWT `sub` 内容 | 获取方式 | 用途 |
|---|---|---|---|
| 用户 Token | 用户 ID | 注册或登录 | 创建会话、查询用户的全部会话 |
| 会话 Token | Session UUID | 创建会话或查询会话列表 | 聊天、流式聊天、消息管理、修改或删除指定会话 |

需要鉴权的接口统一使用以下请求头：

```http
Authorization: Bearer <JWT>
```

默认 JWT 算法为 `HS256`，有效期为 30 天，可通过以下环境变量调整：

```env
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_DAYS=30
```

> 注意：聊天接口必须使用会话 Token，不能直接使用注册或登录返回的用户 Token。

## 2. 通用数据结构

### 2.1 Token

```json
{
  "access_token": "<JWT>",
  "token_type": "bearer",
  "expires_at": "2026-08-23T10:00:00Z"
}
```

### 2.2 Message

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `role` | string | 是 | `user`、`assistant` 或 `system` |
| `content` | string | 是 | 消息正文，长度为 1～3000 个字符 |

请求中的额外消息字段会被忽略。消息内容不能包含空字节或 `<script>...</script>`。

### 2.3 普通业务错误

```json
{
  "detail": "错误说明"
}
```

### 2.4 请求参数校验错误

HTTP 状态码：`422`

```json
{
  "detail": "Validation error",
  "errors": [
    {
      "field": "messages -> 0 -> content",
      "message": "String should have at least 1 character"
    }
  ]
}
```

### 2.5 `request_id`

继承公共响应模型的响应会包含请求追踪 ID：

```json
{
  "request_id": "eac36e99-1901-4490-933e-ae74d5fd7441"
}
```

它可用于关联服务端结构化日志。部分简单字典响应不包含该字段。

## 3. 推荐调用流程

```text
注册或登录
  → 获得用户 Token
  → 创建会话
  → 获得会话 Token
  → 使用会话 Token 调用聊天和消息接口
```

## 4. 基础与运维接口

### 4.1 获取服务信息

```http
GET /
```

无需鉴权。

响应示例：

```json
{
  "name": "FastAPI LangGraph Template",
  "version": "1.0.0",
  "status": "healthy",
  "environment": "development",
  "swagger_url": "/docs",
  "redoc_url": "/redoc"
}
```

默认限流：每分钟 10 次。

### 4.2 综合健康检查

```http
GET /health
```

无需鉴权。该接口会检查数据库连接。

正常响应：HTTP `200`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "components": {
    "api": "healthy",
    "database": "healthy"
  },
  "timestamp": "2026-07-24T10:00:00.000000"
}
```

数据库不可用时返回 HTTP `503`，并将 `status` 设为 `degraded`。

默认限流：每分钟 20 次。

### 4.3 API 简单健康检查

```http
GET /api/v1/health
```

无需鉴权。该接口只返回固定的 API 状态，不检查数据库。

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 4.4 Prometheus 指标

```http
GET /metrics
```

用于 Prometheus 抓取 API 请求、延迟及 LLM 调用等指标。

## 5. 用户认证接口

### 5.1 注册用户

```http
POST /api/v1/auth/register
Content-Type: application/json
```

请求体：

| 字段 | 类型 | 必填 | 约束 |
|---|---|---:|---|
| `email` | string | 是 | 合法电子邮箱 |
| `password` | string | 是 | 8～64 位，必须包含大写字母、小写字母、数字和特殊字符 |
| `username` | string/null | 否 | 最长 50 个字符 |

请求示例：

```json
{
  "email": "user@example.com",
  "password": "Secret123!",
  "username": "张三"
}
```

成功响应：HTTP `200`

```json
{
  "request_id": "eac36e99-1901-4490-933e-ae74d5fd7441",
  "id": 1,
  "email": "user@example.com",
  "username": "张三",
  "token": {
    "access_token": "<用户 JWT>",
    "token_type": "bearer",
    "expires_at": "2026-08-23T10:00:00Z"
  }
}
```

常见错误：

| 状态码 | 原因 |
|---:|---|
| `400` | 邮箱已注册 |
| `422` | 邮箱格式、密码强度或请求字段不符合要求 |
| `429` | 超过限流 |

默认限流：每小时 10 次。

PowerShell 示例：

```powershell
$body = @{
    email = "user@example.com"
    password = "Secret123!"
    username = "张三"
} | ConvertTo-Json

$register = Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/api/v1/auth/register" `
    -ContentType "application/json" `
    -Body $body

$userToken = $register.token.access_token
```

### 5.2 用户登录

```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded
```

该接口接收表单，不接收 JSON。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `email` | string | 是 | 用户邮箱 |
| `password` | string | 是 | 用户密码 |
| `grant_type` | string | 否 | 默认且必须为 `password` |

成功响应：HTTP `200`

```json
{
  "request_id": "eac36e99-1901-4490-933e-ae74d5fd7441",
  "access_token": "<用户 JWT>",
  "token_type": "bearer",
  "expires_at": "2026-08-23T10:00:00Z"
}
```

常见错误：

| 状态码 | 原因 |
|---:|---|
| `400` | `grant_type` 不是 `password` |
| `401` | 邮箱或密码错误 |
| `422` | 表单字段缺失或格式异常 |
| `429` | 超过限流 |

默认限流：每分钟 20 次。

PowerShell 示例：

```powershell
$login = Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/api/v1/auth/login" `
    -ContentType "application/x-www-form-urlencoded" `
    -Body @{
        email = "user@example.com"
        password = "Secret123!"
        grant_type = "password"
    }

$userToken = $login.access_token
```

## 6. 会话管理接口

### 6.1 创建会话

```http
POST /api/v1/auth/session
Authorization: Bearer <用户 JWT>
```

无请求体。

成功响应：HTTP `200`

```json
{
  "request_id": "eac36e99-1901-4490-933e-ae74d5fd7441",
  "session_id": "80a2d1eb-e0ba-41ac-a32d-44e028287a10",
  "name": "",
  "token": {
    "access_token": "<会话 JWT>",
    "token_type": "bearer",
    "expires_at": "2026-08-23T10:00:00Z"
  }
}
```

PowerShell 示例：

```powershell
$session = Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/api/v1/auth/session" `
    -Headers @{ Authorization = "Bearer $userToken" }

$sessionId = $session.session_id
$sessionToken = $session.token.access_token
```

### 6.2 查询当前用户的会话列表

```http
GET /api/v1/auth/sessions
Authorization: Bearer <用户 JWT>
```

成功响应：

```json
[
  {
    "request_id": "eac36e99-1901-4490-933e-ae74d5fd7441",
    "session_id": "80a2d1eb-e0ba-41ac-a32d-44e028287a10",
    "name": "关于 LangGraph 的对话",
    "token": {
      "access_token": "<会话 JWT>",
      "token_type": "bearer",
      "expires_at": "2026-08-23T10:00:00Z"
    }
  }
]
```

接口会为列表中的每个会话生成新的会话 Token。

### 6.3 修改会话名称

```http
PATCH /api/v1/auth/session/{session_id}/name
Authorization: Bearer <该会话 JWT>
Content-Type: application/x-www-form-urlencoded
```

路径参数：

| 参数 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | 要修改的 Session UUID，必须与会话 Token 对应 |

表单参数：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `name` | string | 是 | 新会话名称 |

PowerShell 示例：

```powershell
$updated = Invoke-RestMethod `
    -Method Patch `
    -Uri "http://localhost:8000/api/v1/auth/session/$sessionId/name" `
    -Headers @{ Authorization = "Bearer $sessionToken" } `
    -ContentType "application/x-www-form-urlencoded" `
    -Body @{ name = "LangGraph 学习记录" }
```

成功响应结构与创建会话相同，并返回一个新的会话 Token。

如果路径中的 `session_id` 与 Token 对应的会话不一致，返回 HTTP `403`：

```json
{
  "detail": "Cannot modify other sessions"
}
```

### 6.4 删除会话

```http
DELETE /api/v1/auth/session/{session_id}
Authorization: Bearer <该会话 JWT>
```

路径中的 `session_id` 必须与会话 Token 对应。

成功响应：HTTP `200`

```json
null
```

如果尝试删除其他会话，返回 HTTP `403`。

## 7. 聊天接口

### 7.1 普通聊天

```http
POST /api/v1/chatbot/chat
Authorization: Bearer <会话 JWT>
Content-Type: application/json
```

请求体：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "介绍一下 LangGraph"
    }
  ]
}
```

`messages` 至少包含一条消息。服务会依据会话 Token 自动取得 Session ID 和用户信息，并执行以下流程：

1. 读取该会话的 LangGraph 检查点；
2. 检索当前用户的长期记忆；
3. 调用 LLM；
4. 按需执行搜索或人工询问工具；
5. 返回最终消息并异步更新长期记忆。

成功响应：HTTP `200`

```json
{
  "request_id": "eac36e99-1901-4490-933e-ae74d5fd7441",
  "messages": [
    {
      "role": "assistant",
      "content": "LangGraph 是一个用于构建有状态 Agent 工作流的框架……"
    }
  ]
}
```

如果 Agent 触发 Human-in-the-loop，中断提示也会作为一条 `assistant` 消息返回。下一次使用相同会话 Token 发送用户回答时，图会从中断位置恢复。

常见错误：

| 状态码 | 原因 |
|---:|---|
| `401` | Token 无效或已过期 |
| `404` | Token 对应的会话不存在 |
| `422` | 消息列表或内容不符合 Schema |
| `429` | 超过限流 |
| `500` | LLM、工具或图执行失败 |

默认限流：每分钟 30 次。

PowerShell 示例：

```powershell
$chatBody = @{
    messages = @(
        @{
            role = "user"
            content = "介绍一下 LangGraph"
        }
    )
} | ConvertTo-Json -Depth 5

$answer = Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/api/v1/chatbot/chat" `
    -Headers @{ Authorization = "Bearer $sessionToken" } `
    -ContentType "application/json" `
    -Body $chatBody
```

### 7.2 SSE 流式聊天

```http
POST /api/v1/chatbot/chat/stream
Authorization: Bearer <会话 JWT>
Content-Type: application/json
Accept: text/event-stream
```

请求体与普通聊天接口相同。

响应类型：

```text
text/event-stream
```

每个事件是一行以 `data:` 开头的 JSON：

```text
data: {"request_id":"eac36e99-1901-4490-933e-ae74d5fd7441","content":"Lang","done":false}

data: {"request_id":"eac36e99-1901-4490-933e-ae74d5fd7441","content":"Graph","done":false}

data: {"request_id":"eac36e99-1901-4490-933e-ae74d5fd7441","content":"","done":true}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_id` | UUID | 请求追踪 ID |
| `content` | string | 当前文本片段 |
| `done` | boolean | 是否结束 |

流处理过程中发生错误时，服务仍会通过 SSE 返回结束事件：

```text
data: {"request_id":"...","content":"错误说明","done":true}
```

因此客户端除了检查 HTTP 状态，还需要检查最后一个事件的 `done` 和 `content`。

默认限流：每分钟 20 次。

使用 `curl.exe` 测试：

```powershell
curl.exe -N `
    -X POST "http://localhost:8000/api/v1/chatbot/chat/stream" `
    -H "Authorization: Bearer $sessionToken" `
    -H "Content-Type: application/json" `
    -H "Accept: text/event-stream" `
    -d '{"messages":[{"role":"user","content":"介绍一下 LangGraph"}]}'
```

## 8. 消息历史接口

### 8.1 查询当前会话消息

```http
GET /api/v1/chatbot/messages
Authorization: Bearer <会话 JWT>
```

成功响应：

```json
{
  "request_id": "eac36e99-1901-4490-933e-ae74d5fd7441",
  "messages": [
    {
      "role": "user",
      "content": "介绍一下 LangGraph"
    },
    {
      "role": "assistant",
      "content": "LangGraph 是……"
    }
  ]
}
```

默认限流：每分钟 50 次。

### 8.2 清空当前会话消息

```http
DELETE /api/v1/chatbot/messages
Authorization: Bearer <会话 JWT>
```

该操作会清除当前 Session ID 对应的 LangGraph checkpoint 数据。

成功响应：

```json
{
  "message": "Chat history cleared successfully"
}
```

默认限流：每分钟 50 次。

> 清空会话消息和删除用户长期记忆不是同一操作。该接口不会显式删除 mem0 中已经提取的用户长期记忆。

## 9. 常见状态码

| 状态码 | 含义 |
|---:|---|
| `200` | 请求成功 |
| `400` | 业务参数错误，例如邮箱已注册或 grant type 不支持 |
| `401` | JWT 无效、过期，或登录凭据错误 |
| `403` | 试图修改或删除 Token 所属范围之外的会话 |
| `404` | 用户或会话不存在 |
| `422` | Pydantic 校验失败、Token 格式异常或输入清洗失败 |
| `429` | 请求频率超过限制 |
| `500` | LLM、Agent、数据库操作或工具执行异常 |
| `503` | 综合健康检查发现数据库不可用 |

## 10. 完整 PowerShell 调用示例

以下示例完成“注册 → 创建会话 → 对话 → 查询历史”的完整流程：

```powershell
$baseUrl = "http://localhost:8000"

# 1. 注册并取得用户 Token
$registerBody = @{
    email = "user@example.com"
    password = "Secret123!"
    username = "张三"
} | ConvertTo-Json

$register = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/auth/register" `
    -ContentType "application/json" `
    -Body $registerBody

$userToken = $register.token.access_token

# 2. 创建会话并取得会话 Token
$session = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/auth/session" `
    -Headers @{ Authorization = "Bearer $userToken" }

$sessionToken = $session.token.access_token

# 3. 调用聊天接口
$chatBody = @{
    messages = @(
        @{
            role = "user"
            content = "你好，请介绍一下你的能力。"
        }
    )
} | ConvertTo-Json -Depth 5

$chat = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/chatbot/chat" `
    -Headers @{ Authorization = "Bearer $sessionToken" } `
    -ContentType "application/json" `
    -Body $chatBody

$chat.messages

# 4. 查询会话历史
$history = Invoke-RestMethod `
    -Method Get `
    -Uri "$baseUrl/api/v1/chatbot/messages" `
    -Headers @{ Authorization = "Bearer $sessionToken" }

$history.messages
```

## 11. 默认限流汇总

| 接口 | 默认限制 |
|---|---|
| `GET /` | 10 次/分钟 |
| `GET /health` | 20 次/分钟 |
| `POST /api/v1/auth/register` | 10 次/小时 |
| `POST /api/v1/auth/login` | 20 次/分钟 |
| `POST /api/v1/chatbot/chat` | 30 次/分钟 |
| `POST /api/v1/chatbot/chat/stream` | 20 次/分钟 |
| `GET /api/v1/chatbot/messages` | 50 次/分钟 |
| `DELETE /api/v1/chatbot/messages` | 50 次/分钟 |

这些限制可通过 `.env.<environment>` 中的 `RATE_LIMIT_*` 配置覆盖。当前会话管理接口和 `/api/v1/health` 没有单独的路由限流装饰器。
