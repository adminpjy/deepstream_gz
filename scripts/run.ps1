[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $RootDir ".env"))) {
    throw "Missing .env. Copy .env.example to .env and review its values."
}

$ComposeArgs = @("compose", "up")
if ($Build) { $ComposeArgs += "--build" }
if ($Detach) { $ComposeArgs += "-d" }

Push-Location $RootDir
try {
    & docker @ComposeArgs
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed (exit=$LASTEXITCODE)." }
}
finally {
    Pop-Location
}
