# 配置说明

主配置为 `configs/config.yaml`。相对路径以工程根目录解析，容器部署建议统一使用 `/workspace/...` 绝对路径。字符串支持 `${NAME}` 与 `${NAME:-default}` 环境变量展开；RTSP 密码和数据库连接串只通过环境注入。

## 视频源

`source` 表示单路开发输入；`sources` 非空时优先并支持多路。公共字段：

| 字段 | 含义 |
| --- | --- |
| `camera_id` | 全局唯一业务相机 ID |
| `type` | `file` 或 `rtsp` |
| `path` / `url` | 按类型二选一 |
| `enabled` | 是否构建该路 source bin |
| `nominal_fps` | 用于推理 interval 计算，应与真实源一致 |
| `latency_ms` | RTSP jitter buffer 延迟 |
| `reconnect_interval_sec` | RTSP 重连间隔 |

修改源数量后，`nvstreammux batch-size` 由代码按启用路数设置。文件源采用容器内路径；Windows 路径不能直接写进 YAML。

`scripts/preflight.sh/.ps1` 会在 host 上调用同一配置/资产验证器。若安装了 `ffprobe`，文件源 codec 必须是 H.264 或 H.265/HEVC，且实际 FPS 与 `nominal_fps` 的差异不得超过 `max(0.5 FPS, nominal_fps × 5%)`；这避免错误 interval 在上线后造成推理频率漂移。

## Pipeline 与 Tracker

`pipeline.streammux` 管理 batch 输出尺寸、超时、GPU、时间戳和输入同步；`tiler_width/height` 仅决定合成输出尺寸。当前 PyDS surface 读取、AdaFace TensorRT worker 和编码链按单卡 GPU 0 验证，因此配置预检会明确拒绝任何非零 `gpu_id` 或 nvinfer `gpu-id`，避免形成解码/推理/拷贝跨卡的“部分支持”。

DeepStream 9 的 `nvdcf`、`nvsort`、`deepsort` 共用 `libnvds_nvmultiobjecttracker.so`，真正的算法由 `ll-config-file` YAML 模块决定。`tracker.backend` 只作为可读的意图声明和一致性校验：NvDCF 必须启用 `VisualTracker`，NvSORT 必须启用 `StateEstimator` 且不启用 DCF/DeepSORT ReID，DeepSORT 必须使用 `ReID.reidType=1/3`；不一致会在分配 GPU 前失败。更换实现不改变下游的 `camera_id/track_id/timestamp/bbox` 领域对象。Tracker 配置对分辨率、遮挡、检测 interval 很敏感，生产前必须用现场录像评估 ID switch、轨迹寿命和显存。

## 模型加载规则

- `person` 是主干 PGIE，当前实现要求 `enabled: true`。
- `face.enabled: false` 时不创建人脸 nvinfer。
- `face_recognition.enabled: false` 时不加载 AdaFace；启用它必须同时启用 `face` 和 `database`。
- `behavior.<name>.enabled: false` 时该行为模型不会出现在 Pipeline 中，配置和 engine 路径不做资产要求。
- 所有已启用 nvinfer 的 `unique_id` 必须唯一。
- `runtime.strict_assets: true` 会在创建 GStreamer Pipeline 前检查启用项所需文件。

`models` 节是部署清单，不是隐式开关。实际开关永远是 `person`、`face`、`face_recognition` 和 `behavior.<name>`。

## 推理频率

`inference.person_fps/face_fps/behavior_fps` 必须为正数。程序使用最快已启用源的 `nominal_fps` 换算 DeepStream `interval`：

```text
interval = max(0, round(source_fps / target_fps) - 1)
```

多路源帧率差异较大时，应按摄像机分组部署，避免一个全局 interval 造成慢源过度跳帧。

## 人脸、数据库、截图与输出

`face_recognition` 定义 AdaFace 后端、模型、I/O tensor 名、512 维 embedding、候选帧数量与决策超时。`require_landmarks: true` 会强制五点相似变换；内置 SCRFD 路径应使用 `face.landmark_source: tensor`。`landmark_threshold` 必须与 face nvinfer 的 `pre-cluster-threshold` 完全一致，预检会检查 `output-tensor-meta`、host copy、parser 和单类别声明。`mask` 只为已有自定义 bridge 保留，不是内置 parser 的传输方式。`match_threshold` 与数据库的 `min_similarity` 需要用真实人员库标定，不能把示例值当作验收阈值。

`database.dsn` 留空时从 `dsn_env` 指定的环境变量读取。连接池上下限按进程并发和 PostgreSQL `max_connections` 联合设置。

`snapshot` 决定根目录、格式、JPEG 质量、无人脸人员图的延迟决策和行为截图冷却。业务目录固定为 `person`、`face/know`、`face/unknow`、`behavior`。原图裁剪不叠加框和文字；结果视频的 OSD 独立处理。

`output` 控制 MP4 编码；`encoder: nvidia` 使用 NVMM/NVIDIA 编码器并同时设置周期 I 帧与 IDR，`encoder: x264` 是仅支持 H.264 的 CPU 回退（显式下载到 pinned system memory）。`events_enabled/events_path` 控制供后台消费的 JSONL 事件流。文件源到 EOS 后 muxer 才能完整收尾。停止时应用先发送 EOS；若 Pipeline 拒绝 EOS 会立即退出，等待超时或第二次终止信号会强退。Compose 的 `stop_grace_period` 默认为 95 秒，覆盖 20 秒 Pipeline EOS 和最多 60 秒分析队列/证据落盘收尾，并保留进程退出余量。`runtime.startup_timeout_sec` 控制模型/engine 首次加载等待，`analytics_queue_size` 控制异步业务队列上限。
