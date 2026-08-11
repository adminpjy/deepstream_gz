[CmdletBinding()]
param(
    [switch]$SkipGpuPull,
    [string]$GpuTestImage = "nvidia/cuda:13.0.2-base-ubuntu24.04",
    [string]$EnvFile = ".env",
    [string]$Config = "configs/config.yaml",
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot

function Assert-LastExitCode([string]$Message) {
    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

function Resolve-ProjectPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RootDir $Value))
}

function Import-DotEnv([string]$Path) {
    foreach ($RawLine in Get-Content -LiteralPath $Path) {
        $Line = $RawLine.Trim()
        if (-not $Line -or $Line.StartsWith("#")) { continue }
        if ($Line.StartsWith("export ")) { $Line = $Line.Substring(7) }
        $Match = [regex]::Match($Line, '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
        if (-not $Match.Success) {
            throw "$Path contains an invalid line; only KEY=VALUE is supported: $RawLine"
        }
        $Name = $Match.Groups[1].Value
        $Value = $Match.Groups[2].Value
        if ($Value.Length -ge 2) {
            $First = $Value[0]
            $Last = $Value[$Value.Length - 1]
            if (($First -eq '"' -and $Last -eq '"') -or ($First -eq "'" -and $Last -eq "'")) {
                $Value = $Value.Substring(1, $Value.Length - 2)
            }
        }
        if ($null -eq [Environment]::GetEnvironmentVariable($Name, "Process")) {
            [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
        }
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI is not installed or is not available on PATH."
}

$EnvFile = Resolve-ProjectPath $EnvFile
$Config = Resolve-ProjectPath $Config
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw "Environment file is missing: $EnvFile"
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "Configuration file is missing: $Config"
}

$PythonPrefix = @()
if ($PythonExecutable) {
    if (-not (Get-Command $PythonExecutable -ErrorAction SilentlyContinue)) {
        throw "Host Python executable was not found: $PythonExecutable"
    }
    $PythonCommand = $PythonExecutable
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = "python"
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = "py"
    $PythonPrefix = @("-3")
}
else {
    throw "Host Python was not found. Install Python 3 or pass -PythonExecutable."
}

Import-DotEnv $EnvFile
$SourcePath = Join-Path $RootDir "src"
$ExistingPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
$env:PYTHONPATH = if ($ExistingPythonPath) {
    "$SourcePath$([System.IO.Path]::PathSeparator)$ExistingPythonPath"
} else {
    $SourcePath
}

docker compose version | Out-Null
Assert-LastExitCode "Docker Compose v2 (docker compose) is required."
docker info | Out-Null
Assert-LastExitCode "Docker daemon is unavailable. Start Docker Desktop."

Push-Location $RootDir
try {
    docker compose --env-file $EnvFile config --quiet
    Assert-LastExitCode "docker-compose.yml or $EnvFile could not be parsed."

    Write-Host "[preflight] Validating host configuration, file sources, and enabled model assets..."
    & $PythonCommand @PythonPrefix -m deepstream_ai validate --config $Config
    Assert-LastExitCode "Configuration or asset validation failed. File sources, person nvinfer configuration, and model assets are required."

    & $PythonCommand @PythonPrefix (Join-Path $RootDir "scripts/check-media.py") --config $Config
    Assert-LastExitCode "File source codec or nominal_fps validation failed."

    if ($SkipGpuPull) {
        if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
            throw "nvidia-smi is unavailable."
        }
        nvidia-smi | Out-Null
        Assert-LastExitCode "nvidia-smi failed."
    }
    else {
        Write-Host "[preflight] Validating container GPU access (the first run pulls $GpuTestImage)..."
        docker run --rm --gpus all $GpuTestImage nvidia-smi | Out-Null
        Assert-LastExitCode "The container cannot access the NVIDIA GPU. Check the Windows driver, WSL2, and Docker Desktop GPU integration."
    }
}
finally {
    Pop-Location
}

Write-Host "[preflight] Passed: Docker, Compose, configuration, required assets, file media, and GPU runtime are available."
