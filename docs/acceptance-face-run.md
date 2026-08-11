# Face-enabled 回放验收

以下命令只读取 Compose、日志、数据库和输出文件，不会启动、重启或删除容器、镜像、文件、卷。应在一次完整 `videos/test.mp4` 回放结束后执行；文件源正常结束时 app 容器应为 `exited (0)`，PostgreSQL 应继续为 `running/healthy`。

## 1. 严格配置与资产校验

不要直接执行缺少 `DATABASE_DSN` 的 host validate。用 Compose 渲染后的环境在内存中注入 DSN，不回显该值：

```powershell
$oldDsn = $env:DATABASE_DSN
$oldPythonPath = $env:PYTHONPATH
try {
    $resolved = docker compose --env-file .env config --format json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "Compose config failed." }
    $env:DATABASE_DSN = [string]$resolved.services.app.environment.DATABASE_DSN
    $env:PYTHONPATH = (Resolve-Path src).Path
    python -m deepstream_ai validate --config configs/config.yaml
    if ($LASTEXITCODE -ne 0) { throw "Strict config or asset validation failed." }
}
finally {
    $env:DATABASE_DSN = $oldDsn
    $env:PYTHONPATH = $oldPythonPath
}
```

通过口径：输出 JSON 中 `ok` 为 `true`，person、SCRFD 和 AdaFace 的已启用资产全部存在且语义校验通过。

## 2. 容器状态和完整日志

```powershell
docker compose --env-file .env ps --all

$appId = docker compose --env-file .env ps --all --quiet app
$dbId = docker compose --env-file .env ps --all --quiet postgres
if (-not $appId -or -not $dbId) { throw "App or postgres container is missing." }

$app = (docker inspect $appId | ConvertFrom-Json)[0]
$db = (docker inspect $dbId | ConvertFrom-Json)[0]
if ($app.State.Running -or [int]$app.State.ExitCode -ne 0) {
    throw "File replay did not finish with app exit code 0."
}
if (-not $db.State.Running -or $db.State.Health.Status -ne "healthy") {
    throw "Postgres is not running and healthy."
}

$logs = docker compose --env-file .env logs --no-color --timestamps app 2>&1 | Out-String
$logs
$fatal = "Fatal Python error|Segmentation fault|duplicate output role|Failed to parse bboxes|Traceback \(most recent call last\)|GStreamer error|PipelineError"
if ($logs -match $fatal) { throw "Fatal marker found in app logs." }
foreach ($required in @("person.engine", "face.engine", "sources=1", "(EOS)")) {
    if (-not $logs.Contains($required)) { throw "Required log marker is missing: $required" }
}
```

通过口径：app 为 `exited (0)`；日志同时证明 person/face engine 已加载、Pipeline 进入 PLAYING、收到文件 EOS；不得出现 parser failure、段错误、Python traceback 或 GStreamer fatal error。普通 V4L2 capability probe warning 不能替代上述失败判定，但应保留在验收记录中。

## 3. 输出 MP4 与输入一致性

先保留完整 ffprobe 证据：

```powershell
ffprobe -v error -select_streams v:0 `
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size `
  -of json videos/test.mp4

ffprobe -v error -select_streams v:0 `
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size `
  -of json output/result.mp4
```

再执行机器判定：

```powershell
$inputProbe = ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size -of json videos/test.mp4 | ConvertFrom-Json
$outputProbe = ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size -of json output/result.mp4 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $outputProbe.streams) { throw "Output MP4 is not probeable." }
$inStream = $inputProbe.streams[0]
$outStream = $outputProbe.streams[0]
if ($outStream.codec_name -ne "h264") { throw "Output codec is not H.264." }
if ([int]$outStream.width -ne [int]$inStream.width -or [int]$outStream.height -ne [int]$inStream.height) { throw "Output dimensions differ from input." }
function Convert-FrameRate([string]$Value) {
    $parts = $Value.Split('/')
    if ($parts.Count -ne 2 -or [double]$parts[1] -eq 0) { throw "Invalid frame rate: $Value" }
    return [double]$parts[0] / [double]$parts[1]
}
$inputFps = Convert-FrameRate $inStream.avg_frame_rate
$outputFps = Convert-FrameRate $outStream.avg_frame_rate
$fpsTolerance = [math]::Max(0.5, $inputFps * 0.05)
if ([math]::Abs($outputFps - $inputFps) -gt $fpsTolerance) { throw "Output FPS differs from input." }
$durationTolerance = [math]::Max(1.0, [double]$inputProbe.format.duration * 0.05)
if ([math]::Abs([double]$outputProbe.format.duration - [double]$inputProbe.format.duration) -gt $durationTolerance) { throw "Output duration differs from input by more than 5% or one second." }
if ([int64]$outputProbe.format.size -le 0) { throw "Output MP4 is empty." }
if ($outStream.nb_frames -and [int]$outStream.nb_frames -lt [math]::Floor([int]$inStream.nb_frames * 0.95)) { throw "Output contains fewer than 95% of input frames." }
```

当前固定输入基线为 H.264、1920x1080、10 FPS、223 帧、22.3 秒。输出须可解封装、为 H.264、保持分辨率，时长误差不超过 `max(1 秒, 5%)`；ffprobe 能给出帧数时至少保留 95% 输入帧。

完整顺序解码必须无错误；同时检查周期 IDR，避免 MP4 只能从 0 秒顺序播放、从中间随机跳转却出现缺参考帧的绿色宏块：

```powershell
ffmpeg -hide_banner -v error -i output/result.mp4 -map 0:v:0 -f null NUL
if ($LASTEXITCODE -ne 0) { throw "Full output decode failed." }

$frameProbe = ffprobe -v error -select_streams v:0 -show_frames `
  -show_entries frame=key_frame,pict_type,best_effort_timestamp_time `
  -of json output/result.mp4 | ConvertFrom-Json
$keyFrames = @($frameProbe.frames | Where-Object { [int]$_.key_frame -eq 1 })
$times = @($keyFrames | ForEach-Object { [double]$_.best_effort_timestamp_time })
$maxGap = 0.0
for ($index = 1; $index -lt $times.Count; $index++) {
    $maxGap = [math]::Max($maxGap, $times[$index] - $times[$index - 1])
}
if ($keyFrames.Count -lt 7 -or $maxGap -gt 3.5) {
    throw "Periodic IDR gate failed: count=$($keyFrames.Count), max_gap=$maxGap"
}
```

最后用 ffmpeg 的输入前 seek 路径分别抽取 6、14、20 秒帧，与原片同时间帧做自动像素门禁。该检查不会在工作区创建预览文件；阈值允许检测框和文字 OSD 的正常差异，但能识别大面积解码宏块：

```powershell
@'
import subprocess

import numpy as np

WIDTH, HEIGHT = 480, 270


def fast_frame(path: str, second: int) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-ss", str(second), "-i", path,
        "-frames:v", "1", "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    expected = WIDTH * HEIGHT * 3
    if len(result.stdout) != expected:
        raise SystemExit(f"short frame at {second}s from {path}: {len(result.stdout)}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)


for second in (6, 14, 20):
    source = fast_frame("videos/test.mp4", second)
    output = fast_frame("output/result.mp4", second)
    difference = np.abs(output.astype(np.int16) - source.astype(np.int16))
    mae = float(difference.mean())
    gross = float((difference.max(axis=2) >= 40).mean() * 100.0)
    source_green = (
        (source[:, :, 1].astype(np.int16) > source[:, :, 0].astype(np.int16) + 45)
        & (source[:, :, 1].astype(np.int16) > source[:, :, 2].astype(np.int16) + 45)
        & (source[:, :, 1] > 110)
    )
    output_green = (
        (output[:, :, 1].astype(np.int16) > output[:, :, 0].astype(np.int16) + 45)
        & (output[:, :, 1].astype(np.int16) > output[:, :, 2].astype(np.int16) + 45)
        & (output[:, :, 1] > 110)
    )
    new_green = float((output_green & ~source_green).mean() * 100.0)
    print(f"{second}s: mae={mae:.3f}, gross40={gross:.3f}%, new_green={new_green:.3f}%")
    if mae > 12.0 or gross > 8.0 or new_green > 2.0:
        raise SystemExit(f"fast-seek visual gate failed at {second}s")
'@ | python -
```

当前 NVIDIA 编码配置的验收预期为 0、3、6、…、21 秒共 8 个关键帧，最大间隔 3 秒。还应人工抽看至少一个识别后的帧，身份标签应为 `person <track> unknown/worker...`，不得出现 12 位以上的指针地址。

## 4. JSONL events

```powershell
@'
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import UUID

path = Path("output/events.jsonl")
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit("events.jsonl is missing or empty")

required = {"event_id", "event_type", "camera_id", "track_id", "timestamp", "payload", "attributes"}
allowed = {"person", "face", "identity", "behavior", "track_ended", "snapshot"}
counts = Counter()
event_ids = set()
identity_keys = set()
snapshot_paths = []
for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON at line {line_number}: {exc}") from exc
    missing = required - event.keys()
    if missing:
        raise SystemExit(f"line {line_number} missing keys: {sorted(missing)}")
    event_id = UUID(event["event_id"])
    if event_id in event_ids:
        raise SystemExit(f"duplicate event_id at line {line_number}: {event_id}")
    event_ids.add(event_id)
    datetime.fromisoformat(event["timestamp"])
    if not event["camera_id"] or event["track_id"] == "":
        raise SystemExit(f"invalid camera_id/track_id at line {line_number}")
    event_type = event["event_type"]
    if event_type not in allowed:
        raise SystemExit(f"unknown event_type at line {line_number}: {event_type}")
    counts[event_type] += 1
    if event_type == "identity":
        identity_key = (event["camera_id"], str(event["track_id"]))
        if identity_key in identity_keys:
            raise SystemExit(f"duplicate semantic identity at line {line_number}: {identity_key}")
        identity_keys.add(identity_key)
    if event_type == "snapshot":
        snapshot_paths.append(str(event["payload"].get("path", "")))

for required_type in ("person", "face", "identity", "snapshot"):
    if counts[required_type] == 0:
        raise SystemExit(f"face-enabled replay produced no {required_type} events")
if counts["behavior"] != 0:
    raise SystemExit("behavior events exist while all behavior models are disabled")
for raw_path in snapshot_paths:
    if raw_path.startswith("/workspace/"):
        candidate = Path(raw_path.removeprefix("/workspace/"))
    else:
        candidate = Path(raw_path)
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise SystemExit(f"snapshot event points to missing/empty file: {raw_path}")
print("event_counts=" + json.dumps(counts, sort_keys=True))
print(
    f"unique_event_ids={len(event_ids)} identities={len(identity_keys)} "
    f"snapshot_references={len(snapshot_paths)}"
)
'@ | python -
```

通过口径：每行都是完整 JSON；`event_id` 唯一且合法；字段、时间戳和事件类型合法；本次 face-enabled 素材至少产生 person、face、identity、snapshot 各一条；identity 按 `(camera_id, track_id)` 只能发布一次；每个 snapshot event 指向真实非空文件。当前 behavior 模型全部关闭，因此 behavior event 必须为 0。

## 5. Snapshot 文件

```powershell
$snapshotRoot = Resolve-Path output/snapshot -ErrorAction Stop
$images = @(Get-ChildItem $snapshotRoot -Recurse -File | Where-Object { $_.Extension -in @(".jpg", ".jpeg", ".png") })
$faces = @($images | Where-Object { $_.FullName -match '[\\/]face[\\/](know|unknow)[\\/]' })
$temporary = @(Get-ChildItem $snapshotRoot -Recurse -File | Where-Object { $_.Name -match '\.tmp($|\.)' })
if ($images.Count -eq 0) { throw "No snapshots were produced." }
if ($faces.Count -eq 0) { throw "Face-enabled replay produced no known/unknown face snapshot." }
if (@($images | Where-Object Length -le 0).Count -ne 0) { throw "An empty snapshot exists." }
if ($temporary.Count -ne 0) { throw "Temporary snapshot files remain after shutdown." }
foreach ($image in $images) {
    $probe = ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height -of json $image.FullName | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not $probe.streams -or [int]$probe.streams[0].width -le 0 -or [int]$probe.streams[0].height -le 0) {
        throw "Snapshot is not decodable: $($image.FullName)"
    }
}
$images | Group-Object { Split-Path $_.DirectoryName -Leaf } | Select-Object Name,Count
```

人脸库为空时，`face/unknow` 至少应有一张，`face/know` 可以为空；`person` 只保存全程没有人脸的 track，因此不要求每个 person track 都有 person snapshot；behavior 模型关闭时 `behavior` 目录可以不存在。截图必须非空且没有遗留临时文件。

可选地只读确认人脸库数量：

```powershell
docker compose --env-file .env exec -T postgres sh -lc 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SELECT count(*) FROM t_worker_face_vector;"'
```
