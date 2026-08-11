[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
Push-Location $RootDir
try {
    docker compose config --quiet
    if ($LASTEXITCODE -ne 0) { throw "Compose configuration validation failed." }
    & docker compose run --rm --no-deps app python3 -m pytest -q @PytestArgs
    if ($LASTEXITCODE -ne 0) { throw "Tests failed (exit=$LASTEXITCODE)." }
}
finally {
    Pop-Location
}
