# Auth 与 Chatbot 接口文档

基础地址：

```text
http://base_url:8090
```

Token 说明：

- 用户 Token：注册或登录后获取，用于创建、查询会话。
- 会话 Token：创建或查询会话后获取，用于 Chatbot 和删除会话。

## Auth

### 1. 注册

API 方式：HTTP POST

API 路径：

```text
http://base_url:8090/api/v1/auth/register
```

请求参数/header：

```text
Content-Type: application/json
```

请求体：

```json
{
  "email": "developer@example.com",
  "password": "StrongPass1!",
  "username": "张三"
}
```

响应体：

```json
{
  "request_id": "a2073856-ae58-4987-a268-d50851682bad",
  "id": 12,
  "email": "developer@example.com",
  "username": "张三",
  "token": {
    "access_token": "用户Token",
    "token_type": "bearer",
    "expires_at": "2026-08-27T10:00:00Z"
  }
}
```

说明：密码长度为 8～64 位，且必须包含大写字母、小写字母、数字和特殊字符。

### 2. 登录

API 方式：HTTP POST

API 路径：

```text
http://base_url:8090/api/v1/auth/login
```

请求参数/header：

```text
Content-Type: application/x-www-form-urlencoded
```

请求体：

```text
email=developer@example.com
password=StrongPass1!
grant_type=password
```

响应体：

```json
{
  "request_id": "4aa1d876-1102-459e-bd5c-34e59cc2d35c",
  "access_token": "用户Token",
  "token_type": "bearer",
  "expires_at": "2026-08-27T10:05:00Z"
}
```

### 3. 创建会话

API 方式：HTTP POST

API 路径：

```text
http://base_url:8090/api/v1/auth/session
```

请求参数/header：

```text
Authorization: Bearer 用户Token
```

请求体：

```json
{}
```

响应体：

```json
{
  "request_id": "dd386580-ed14-41df-9079-84d13fdd2b06",
  "session_id": "80a2d1eb-e0ba-41ac-a32d-44e028287a10",
  "name": "",
  "token": {
    "access_token": "会话Token",
    "token_type": "bearer",
    "expires_at": "2026-08-27T10:10:00Z"
  }
}
```

### 4. 查询会话

API 方式：HTTP GET

API 路径：

```text
http://base_url:8090/api/v1/auth/sessions
```

请求参数/header：

```text
Authorization: Bearer 用户Token
```

请求体：

```json
{}
```

响应体：

```json
[
  {
    "request_id": "b921300a-f085-46b2-95c0-fbcbf05f144e",
    "session_id": "80a2d1eb-e0ba-41ac-a32d-44e028287a10",
    "name": "ANN模型查询",
    "token": {
      "access_token": "会话Token",
      "token_type": "bearer",
      "expires_at": "2026-08-27T10:15:00Z"
    }
  }
]
```

### 5. 删除会话

API 方式：HTTP DELETE

API 路径：

```text
http://base_url:8090/api/v1/auth/session/{session_id}
```

请求参数/header：

```text
Authorization: Bearer 会话Token
```

请求体：

```json
{}
```

响应体：

```json
null
```

说明：路径中的 `session_id` 必须与会话 Token 对应。

## Chatbot

### 1. Chat

API 方式：HTTP POST

API 路径：

```text
http://base_url:8090/api/v1/chatbot/chat
```

请求参数/header：

```text
Authorization: Bearer 会话Token
Content-Type: application/json
```

请求体：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "查询ANN模型"
    }
  ]
}
```

响应体：

```json
{
  "request_id": "90fe7be5-9110-48c2-9655-99795d19c99a",
  "messages": [
    {
      "role": "user",
      "content": "查询ANN模型"
    },
    {
      "role": "assistant",
      "content": "ANN模型的相关回答"
    }
  ]
}
```

### 2. 流式 Chat

API 方式：HTTP POST（event-stream）

API 路径：

```text
http://base_url:8090/api/v1/chatbot/chat/stream
```

请求参数/header：

```text
Authorization: Bearer 会话Token
Content-Type: application/json
Accept: text/event-stream
```

请求体：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "查询ANN模型"
    }
  ]
}
```

普通消息响应体：

```text
data: {"request_id":"4f1a60b8-e4c5-47e6-8bd8-b1732818246c","event":"message","content":"回答内容","done":false}

data: {"request_id":"4f1a60b8-e4c5-47e6-8bd8-b1732818246c","event":"done","content":"","done":true}
```

查询澄清 HITL 响应体：

```text
hitl: {"request_id":"4f1a60b8-e4c5-47e6-8bd8-b1732818246c","event":"hitl","content":"","hitl_type":"question_clarification","title":"请选择知识库目录并补充问题","directories":[{"title":"Revin","uri":"viking://resources/Revin"}],"done":false}

data: {"request_id":"4f1a60b8-e4c5-47e6-8bd8-b1732818246c","event":"done","content":"","done":true}
```

查询澄清 HITL 恢复请求体：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "{\"system_uri\":\"viking://resources/Revin\",\"question\":\"查询ANN模型的架构和应用场景\"}"
    }
  ]
}
```

身份 HITL 响应体：

```text
hitl: {"request_id":"4f1a60b8-e4c5-47e6-8bd8-b1732818246c","event":"hitl","content":"","hitl_type":"role_clarification","title":"请选择用户身份","options":[{"title":"产品经理","value":"product_manager"},{"title":"开发","value":"developer"},{"title":"新入职员工","value":"new_employee"}],"done":false}

data: {"request_id":"4f1a60b8-e4c5-47e6-8bd8-b1732818246c","event":"done","content":"","done":true}
```

身份 HITL 恢复请求体：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "{\"role\":\"developer\"}"
    }
  ]
}
```

错误响应体：

```text
data: {"request_id":"4f1a60b8-e4c5-47e6-8bd8-b1732818246c","event":"error","content":"错误信息","done":true}
```

说明：

- 普通消息使用 `data:`。
- HITL 使用 `hitl:`。
- HITL 恢复必须继续使用同一个会话 Token。
- `request_id` 不需要在恢复请求中传回。
