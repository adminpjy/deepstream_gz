# Production recognition service upgrade

This document describes the production RTSP path introduced on branch
`feature/production-session-multigpu`. The existing file/task pipeline and the
tuned PeopleNet -> NvDCF -> SCRFD -> AdaFace behavior remain available and are
not replaced.

## 1. Runtime architecture

```text
CPU static-change service / test Web / formal system
                    |
             REST control plane
                    |
        ProductionRecognitionService
                    |
        least-active READY GPU
          /         |         \
 GPU Worker 0  GPU Worker 1  GPU Worker N
  persistent     persistent     persistent
  CUDA mapped    CUDA mapped    CUDA mapped
  models warm    models warm    models warm
       |              |              |
  dynamic RTSP sessions (attach / detach only)
       |
 tuned core analytics (always on)
       |
 independent optional scenario processors
       |
 RecognitionEvent -> ResultPublisher
```

Each physical GPU owns one long-lived worker process. The supervisor starts all
workers before the HTTP service reports ready. Each worker receives
`CUDA_VISIBLE_DEVICES=<physical gpu id>` so the existing DeepStream configs
continue to use the already validated logical `gpu-id=0` contract. No tracker,
face detector, face-recognition threshold or continuity rule is changed for GPU
selection.

A production session does **not** start a new DeepStream process. It attaches one
RTSP SourceBin to an already PLAYING worker Pipeline. Stopping a session removes
that source, so decode and inference compute for the camera is released while
the worker, CUDA context, TensorRT models, tracker and face-recognition runtime
remain resident.

### Production batch engine isolation

The current PeopleNet serialized engine is a batch-1 asset used by the existing
task pipeline. A multi-stream production worker needs a larger PGIE batch. The
production builder therefore materializes an isolated engine under:

```text
${PRODUCTION_OUTPUT_ROOT}/gpu-<id>/.engines/
```

The legacy model engine under `/workspace/models` is never overwritten. The
production engine filename includes worker capacity and a model/config
fingerprint. First deployment may build the production engine during worker
startup; later service starts reuse the cached engine. HTTP readiness is not
published until workers have completed their startup path.

## 2. Core and optional recognition

Core analytics are mandatory and do not appear as disable switches in the REST
contract:

- person detection
- person tracking / continuity
- face detection and tracking association
- AdaFace recognition

Optional analytics are selected per session:

- `smoking`
- `eating`
- `drinking`
- `leftObject`
- `largeObjectMoving` (reserved; explicit `FEATURE_NOT_IMPLEMENTED`)

Smoking/eating/drinking TensorRT SGIEs are loaded only when their deployed
assets are available. Once loaded, they stay resident in every GPU worker. A
per-source inference gate prevents disabled cameras from entering that SGIE, so
"model resident" does not mean "every camera executes every model".

Each scene has an independent processor. Processors only consume the normalized
frame/track contract and publish `RecognitionEvent`; they do not call each
other.

## 3. Left-object detection

The left-object path intentionally has no new AI model and no person-object
association.

1. The CPU static server, or the test Web page, uploads the camera's last stable
   no-person image before GPU activation.
2. The GPU session runs normal person/face recognition while people are present.
3. After the configured no-person interval, the `LeftObjectProcessor` compares
   the baseline with several recent no-person frames.
4. OpenCV processing performs luminance normalization, blur, absolute
   difference, thresholding, morphology and connected-component filtering.
5. A change must persist for multiple samples before `LEFT_OBJECT` is emitted.
6. `before.jpg`, `after.jpg` and `diff.jpg` evidence are written under the
   session directory.
7. The baseline is **not automatically replaced** when an alarm is detected.

Baseline upload:

```http
POST /api/v1/recognition/cameras/{cameraId}/baseline
Content-Type: image/jpeg

<binary image>
```

## 4. REST API

### Capabilities

```http
GET /api/v1/recognition/capabilities
```

The response states whether smoking/eating/drinking assets are actually
available. A requested but unavailable feature is rejected before source
allocation.

### GPU status

```http
GET /api/v1/recognition/gpus
GET /api/v1/recognition/service
```

### Start a session

```http
POST /api/v1/recognition/sessions/start
Content-Type: application/json
```

Example:

```json
{
  "cameraId": "room-a-01",
  "streamUrl": "rtsp://10.10.10.20/live",
  "nominalFps": 30,
  "features": {
    "smoking": true,
    "eating": false,
    "drinking": true,
    "leftObject": true,
    "largeObjectMoving": false
  },
  "exitPolicy": {
    "personAbsentSeconds": 30
  },
  "leftObject": {
    "pixelThreshold": 28,
    "minAreaRatio": 0.0015,
    "minComponentAreaRatio": 0.00035,
    "confirmFrames": 3,
    "maxRecentFrames": 8
  },
  "context": {
    "cameraName": "A机房入口"
  }
}
```

The caller does not select a GPU. The service selects the READY worker with the
fewest active sessions. A camera may have only one active production session.

### Session status and list

```http
GET /api/v1/recognition/sessions
GET /api/v1/recognition/sessions/{sessionId}
GET /api/v1/recognition/sessions/{sessionId}/preview.jpg
```

### Stop

```http
POST /api/v1/recognition/sessions/{sessionId}/stop
Content-Type: application/json

{"reason":"formal_system_requested"}
```

or:

```http
POST /api/v1/recognition/cameras/{cameraId}/stop
```

A manual stop removes the stream immediately. The automatic no-person stop runs
the left-object post-check first when `leftObject=true`, then detaches the RTSP
source. It does not terminate the GPU worker.

## 5. Result publishing boundary

Recognition and transport are isolated:

```text
core/scenario recognition
        |
 RecognitionEvent
        |
 ResultPublisher
     /       \
 local JSONL  HTTP adapter
```

Local events are always journaled under:

```text
${PRODUCTION_OUTPUT_ROOT}/recognition-events.jsonl
```

Set these variables to enable the generic REST publisher:

```text
RESULT_PUBLISH_URL=
RESULT_PUBLISH_TOKEN=
RESULT_PUBLISH_TIMEOUT_SEC=3
RESULT_PUBLISH_MAX_ATTEMPTS=3
RESULT_PUBLISH_QUEUE_SIZE=1024
```

Formal-system field mapping belongs in `HttpResultPublisher.to_external_payload`
or a replacement `ResultPublisher`. Scenario code, person/face recognition and
session management must not contain formal-system URLs, auth headers or DTO
fields.

HTTP publishing is queued and bounded so a slow formal endpoint does not block
the DeepStream streaming thread. Failed final deliveries are written to the
dead-letter JSONL file.

## 6. Multi-GPU configuration

By default the service discovers all GPUs visible in the container. To pin a
subset:

```text
SERVICE_GPU_IDS=0,1
```

The IDs are the GPU indices visible to `nvidia-smi` inside the application
container.

Per-worker stream capacity is an admission-control and PGIE batch setting:

```text
SERVICE_SESSIONS_PER_GPU=16
```

Do not raise this value only because GPU memory is free. Validate the final value
with the real camera codec/resolution, PeopleNet/NvDCF/face workload and enabled
optional models. A capacity change creates a different production PGIE engine
cache and therefore takes effect on the next worker restart.

## 7. Test Web page

The existing local-file test flow remains on `/api/tasks` unchanged. When the
RTSP tab is selected, the page uses the production session API and shows:

- fixed-on core recognition
- smoking/eating/drinking switches, disabled automatically if assets are missing
- left-object switch and optional baseline upload
- large-object-moving reserved/disabled
- person-absence auto-exit seconds
- production GPU/session cards with preview and stop controls

## 8. Production files and diagnostics

```text
${PRODUCTION_OUTPUT_ROOT}/
  control/                  private Unix worker sockets
  gpu-0/
    worker.log
    worker-status.json
    .engines/               isolated production PGIE cache
    .runtime/nvinfer/
  gpu-1/
    ...
  baselines/<cameraId>/
  sessions/<sessionId>/
    status.json
    preview.jpg
    left-object/
      before.jpg
      after.jpg
      diff.jpg
  alarm-evidence/
  recognition-events.jsonl
  result-dead-letter.jsonl
```

## 9. Required acceptance checks on the GPU host

Before merging this branch to the production line, run on the actual target GPU
server:

1. Run the existing repository test suite and compare current person/face test
   artifacts with `main`.
2. Start the service with at least two GPUs and verify all configured workers are
   READY concurrently.
3. Start two RTSP sessions and verify least-active GPU allocation.
4. Stop and restart the same camera repeatedly without worker/model restart.
5. Verify core person tracking, face extraction and AdaFace results against the
   current tuned baseline.
6. Enable only smoking and verify eating/drinking SGIE gates do not infer for the
   source; repeat for each optional feature.
7. Upload a baseline, leave an object, wait for person absence, and verify
   `LEFT_OBJECT` plus before/after/diff evidence before automatic detach.
8. Verify unchanged lighting/scene does not create a left-object event.
9. Make the external result endpoint unavailable and verify recognition remains
   live while events go to dead-letter output.
10. Load-test the intended `SERVICE_SESSIONS_PER_GPU` using real stream codecs,
    resolution and GOP settings; record attach-to-first-result P50/P95.

The implementation deliberately keeps GPU-host validation separate from unit
checks because DeepStream/GStreamer/TensorRT runtime correctness cannot be
established on a non-GPU development host.
