[CmdletBinding()]
param(
    [string]$Archive = "output/deepstream-ai-platform.tar",
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot

function Resolve-ProjectPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RootDir $Value))
}

$Archive = Resolve-ProjectPath $Archive
$EnvFile = Resolve-ProjectPath $EnvFile
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw "Environment file does not exist: $EnvFile"
}

Push-Location $RootDir
try {
    $ComposeJson = (& docker compose --env-file $EnvFile config --format json | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "Compose configuration could not be rendered with $EnvFile." }
    try {
        $ComposeConfig = $ComposeJson | ConvertFrom-Json
        $Image = [string]$ComposeConfig.services.app.image
    }
    catch {
        throw "Could not resolve services.app.image from Compose configuration: $($_.Exception.Message)"
    }
    if (-not $Image) { throw "Compose services.app.image is empty." }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Archive) | Out-Null
    docker compose --env-file $EnvFile build app
    if ($LASTEXITCODE -ne 0) { throw "Image build failed: $Image" }
    docker image inspect $Image | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "The image built by Compose is unavailable: $Image" }
    docker save --output $Archive $Image
    if ($LASTEXITCODE -ne 0) { throw "docker save failed: $Image" }
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
    "$Hash  $([System.IO.Path]::GetFileName($Archive))" |
        Set-Content -Encoding ascii -LiteralPath "$Archive.sha256"
}
finally {
    Pop-Location
}

Write-Host "Exported Compose services.app.image $Image -> $Archive"
