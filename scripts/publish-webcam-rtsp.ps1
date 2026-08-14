param(
    [string]$Device = "Integrated Webcam",
    [string]$Url = "rtsp://10.101.176.41:8554/camera",
    [int]$BitrateKbps = 1500,
    [int]$Fps = 30,
    [string]$Resolution = "1280x720"
)

$ErrorActionPreference = "Stop"

$ffmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpegCommand) {
    throw "ffmpeg was not found. Install ffmpeg and add it to PATH."
}

$BufferKbps = [Math]::Max(250, [int]($BitrateKbps / 2))

Write-Host "Publishing $Device -> $Url"
Write-Host "Resolution=$Resolution FPS=$Fps Bitrate=${BitrateKbps}k Buffer=${BufferKbps}k"

$ffmpegArgs = @(
    "-hide_banner",
    "-f", "dshow",
    "-vcodec", "mjpeg",
    "-video_size", $Resolution,
    "-framerate", $Fps.ToString(),
    "-i", "video=$Device",
    "-an",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",
    "-b:v", "${BitrateKbps}k",
    "-maxrate", "${BitrateKbps}k",
    "-bufsize", "${BufferKbps}k",
    "-g", $Fps.ToString(),
    "-keyint_min", $Fps.ToString(),
    "-sc_threshold", "0",
    "-bf", "0",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    $Url
)

& $ffmpegCommand.Source @ffmpegArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Error "ffmpeg exited with code $exitCode"
}

exit $exitCode
