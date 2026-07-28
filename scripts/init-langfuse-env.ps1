[CmdletBinding()]
param(
    [switch]$Force
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $projectRoot ".env.langfuse"

if ((Test-Path -LiteralPath $outputPath) -and -not $Force) {
    throw ".env.langfuse already exists. Use -Force to regenerate it."
}

function New-HexSecret {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ByteCount
    )

    $bytes = New-Object byte[] $ByteCount
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return ([BitConverter]::ToString($bytes) -replace "-", "").ToLowerInvariant()
}

$nextAuthSecret = New-HexSecret -ByteCount 32
$salt = New-HexSecret -ByteCount 32
$encryptionKey = New-HexSecret -ByteCount 32
$postgresPassword = New-HexSecret -ByteCount 24
$clickhousePassword = New-HexSecret -ByteCount 24
$redisPassword = New-HexSecret -ByteCount 24
$minioPassword = New-HexSecret -ByteCount 24

$content = @"
LANGFUSE_PORT=3001
MINIO_API_PORT=9190
MINIO_CONSOLE_PORT=9191
OBSERVABILITY_NETWORK=observability

NEXTAUTH_URL=http://localhost:3001
NEXTAUTH_SECRET=$nextAuthSecret
SALT=$salt
ENCRYPTION_KEY=$encryptionKey

POSTGRES_PASSWORD=$postgresPassword
CLICKHOUSE_PASSWORD=$clickhousePassword
REDIS_AUTH=$redisPassword
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=$minioPassword

TELEMETRY_ENABLED=false
LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false
"@

[IO.File]::WriteAllText(
    $outputPath,
    $content,
    [Text.UTF8Encoding]::new($false)
)

Write-Host "Generated .env.langfuse. Secret values were not printed."
