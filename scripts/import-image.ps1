[CmdletBinding()]
param([string]$Archive = "output/deepstream-ai-platform.tar")

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Archive)) {
    throw "Image archive does not exist: $Archive"
}

$ChecksumPath = "$Archive.sha256"
if (Test-Path -LiteralPath $ChecksumPath) {
    $Expected = ((Get-Content -LiteralPath $ChecksumPath -Raw).Trim() -split '\s+')[0]
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash
    if (-not $Actual.Equals($Expected, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "SHA256 verification failed: $Archive"
    }
}
else {
    Write-Warning "Checksum file was not found: $ChecksumPath"
}

docker load --input $Archive
if ($LASTEXITCODE -ne 0) { throw "docker load failed." }
