param(
    [switch]$NoPull,
    [switch]$Build,
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

    Write-Step "Check Docker Desktop"
    # Docker Desktop writes harmless capability warnings (for example blkio
    # support notices) to stderr even when `docker info` succeeds. With
    # $ErrorActionPreference='Stop', PowerShell can promote that stderr text to
    # a terminating NativeCommandError. Run the readiness probe through cmd.exe
    # so stdout/stderr are discarded and only docker's process exit code decides
    # whether the engine is ready.
    & cmd.exe /d /c "docker info >nul 2>&1"
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop is not ready. Please start Docker Desktop and wait until the engine is running."
    }
    Write-Host "Docker is ready." -ForegroundColor Green

    if (-not $NoPull) {
        if (Get-Command git -ErrorAction SilentlyContinue) {
            Write-Step "Pull latest code"
            $branch = (& git branch --show-current).Trim()
            $origin = (& git remote get-url origin).Trim()
            Write-Host "Git branch: $branch"
            Write-Host "Git origin: $origin"
            Invoke-Checked -Command "git" -Arguments @("pull", "--ff-only")
        }
        else {
            Write-Host "Git not found, skip git pull." -ForegroundColor Yellow
        }
    }

    Write-Step "Stop existing services"
    Invoke-Checked -Command "docker" -Arguments @("compose", "down")

    Write-Step ($(if ($Build) { "Build image and start services" } else { "Start services (reuse existing image)" }))
    $upArgs = @("compose", "up")
    if ($Build) {
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
    Write-Host "  .\start.ps1              # pull + restart, reuse existing image" -ForegroundColor DarkGray
    Write-Host "  .\start.ps1 -NoPull      # restart without git pull" -ForegroundColor DarkGray
    Write-Host "  .\start.ps1 -Build       # rebuild Docker image, then restart" -ForegroundColor DarkGray
    Write-Host "  .\start.ps1 -Logs        # restart then follow app logs" -ForegroundColor DarkGray

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
