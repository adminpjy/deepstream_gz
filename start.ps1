param(
    [switch]$NoPull,
    [switch]$NoBuild,
    [switch]$Logs
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

try {
    Write-Host "DeepStream AI Platform - Start / Restart" -ForegroundColor Green
    Write-Host "Project: $PSScriptRoot"

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker command not found. Please start Docker Desktop first."
    }

    Invoke-Checked -Command "docker" -Arguments @("info")

    if (-not $NoPull) {
        if (Get-Command git -ErrorAction SilentlyContinue) {
            Write-Step "Pull latest code"
            Invoke-Checked -Command "git" -Arguments @("pull", "--ff-only")
        }
        else {
            Write-Host "Git not found, skip git pull." -ForegroundColor Yellow
        }
    }

    Write-Step "Stop existing services"
    Invoke-Checked -Command "docker" -Arguments @("compose", "down")

    Write-Step "Start services"
    $upArgs = @("compose", "up")
    if (-not $NoBuild) {
        $upArgs += "--build"
    }
    $upArgs += @("-d")
    Invoke-Checked -Command "docker" -Arguments $upArgs

    Write-Step "Service status"
    Invoke-Checked -Command "docker" -Arguments @("compose", "ps")

    Write-Host ""
    Write-Host "Started successfully." -ForegroundColor Green
    Write-Host "Web console: http://127.0.0.1:8080" -ForegroundColor Green
    Write-Host ""
    Write-Host "Common usage:" -ForegroundColor DarkGray
    Write-Host "  .\start.ps1              # pull + restart + build" -ForegroundColor DarkGray
    Write-Host "  .\start.ps1 -NoPull      # restart without git pull" -ForegroundColor DarkGray
    Write-Host "  .\start.ps1 -NoBuild     # restart without rebuilding image" -ForegroundColor DarkGray
    Write-Host "  .\start.ps1 -Logs        # start then follow app logs" -ForegroundColor DarkGray

    if ($Logs) {
        Write-Step "Follow app logs (Ctrl+C to exit logs; containers keep running)"
        & docker compose logs -f app
    }
}
catch {
    Write-Host ""
    Write-Host "Start/restart failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Run 'docker compose logs app postgres' for details." -ForegroundColor Yellow
    exit 1
}
