# Langfuse 本地部署

项目使用独立的 Langfuse v3 Compose，不复用业务 PostgreSQL、Redis 或
Grafana。Langfuse 页面使用 `3001` 端口，避免与项目 Grafana 的 `3000`
端口冲突。

## 1. 生成本地密钥

在项目根目录运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\init-langfuse-env.ps1
```

脚本会生成被 Git 忽略的 `.env.langfuse`，其中包含随机生成的数据库、
Redis、MinIO 和 Langfuse 加密密钥。`ExecutionPolicy Bypass` 只对本次
子进程有效，不会修改系统执行策略。

## 2. 启动 Langfuse

```powershell
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml up -d
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml ps
```

首次启动需要下载镜像并执行数据库迁移。容器启动后打开：

- Langfuse：<http://localhost:3001>
- MinIO 控制台：<http://localhost:9191>

首次进入 Langfuse 时注册本地管理员，创建项目，然后在项目设置中创建
API Keys。

## 3. 接入当前 FastAPI 项目

把 Langfuse 项目设置页生成的公钥和密钥写入当前运行环境对应的
`.env.development`：

### FastAPI 直接运行在 Windows 主机

```dotenv
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-替换为本地项目公钥
LANGFUSE_SECRET_KEY=sk-lf-替换为本地项目密钥
```

### FastAPI 运行在 Docker 容器

```dotenv
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_PUBLIC_KEY=pk-lf-替换为本地项目公钥
LANGFUSE_SECRET_KEY=sk-lf-替换为本地项目密钥
```

两个 Compose 通过外部 `observability` 网络互通。首次启动前创建网络：

```powershell
docker network inspect observability *> $null
if ($LASTEXITCODE -ne 0) {
    docker network create observability
}
```

修改后重启 FastAPI 服务。当前项目会在启动时调用 Langfuse
`auth_check()`，认证成功后才会启用 LangChain/LangGraph 回调跟踪。

## 4. 常用运维命令

查看日志：

```powershell
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml logs -f langfuse-web langfuse-worker
```

停止服务但保留数据：

```powershell
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml down
```

拉取 Langfuse v3 最新镜像并升级：

```powershell
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml pull
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml up -d
```

删除容器及全部 Langfuse 本地数据：

```powershell
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml down -v
```

最后一条命令会永久删除追踪记录、用户、项目及对象存储数据，只应在确认
不再需要数据时执行。

## 5. 暴露给局域网时

当前配置以本机访问为默认目标。如果从另一台电脑访问，应同时把
`.env.langfuse` 中的 `NEXTAUTH_URL` 改为主机实际地址，并把 Compose 中
MinIO 的外部地址从 `localhost` 改为该地址；同时配置防火墙和反向代理。
