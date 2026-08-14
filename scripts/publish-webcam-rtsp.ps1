param(
    [string]$Device = "Integrated Webcam",
    [string]$Url = "rtsp://10.101.176.41:8554/camera",
    [int]$BitrateKbps = 1500,
    [int]$Fps = 30,
    [string]$Resolution = "1280x720"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg 未找到，请先安装并加入 PATH。"
}

$BufferKbps = [Math]::Max(250, [int]($BitrateKbps / 2))

Write-Host "Publishing $Device -> $Url"
Write-Host "Resolution=$Resolution FPS=$Fps Bitrate=${BitrateKbps}k Buffer=${BufferKbps}k"

& ffmpeg `
    -hide_banner `
    -f dshow `
    -vcodec mjpeg `
    -video_size $Resolution `
    -framerate $Fps `
    -i "video=$Device" `
    -an `
    -c:v libx264 `
    -preset veryfast `
    -tune zerolatency `
    -pix_fmt yuv420p `
    -b:v "${BitrateKbps}k" `
    -maxrate "${BitrateKbps}k" `
    -bufsize "${BufferKbps}k" `
    -g $Fps `
    -keyint_min $Fps `
    -sc_threshold 0 `
    -bf 0 `
    -f rtsp `
    -rtsp_transport tcp `
    $Url

exit $LASTEXITCODE
