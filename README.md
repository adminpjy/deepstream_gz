# DeepStream AI Platform

一个配置驱动、面向多路视频的 NVIDIA DeepStream 工业智能分析初版工程。主视频链路由 GStreamer/DeepStream 完成硬件解码、批处理、TensorRT 推理、跟踪、OSD 与硬件编码；Python 只负责建图、元数据路由、人脸融合、向量检索、事件与截图业务，不使用 Python 循环或 OpenCV 充当主 Pipeline。

## 当前交付边界

工程代码、容器、pgvector、配置、脚本、测试入口和部署文档均已就位。当前工作区的 `videos/test.mp4` 已转码为 NVDEC 兼容的 H.264，原 MPEG-4 Part 2 文件保留为 `videos/test-original-mpeg4.mp4`；Git 默认忽略视频，全新 clone 必须自行提供获授权的测试片，否则预检会按设计失败。仓库**不分发模型权重或 TensorRT engine**。这是许可、安全和可复现性要求，不是静默降级：

- 人员检测是主干组件，必须提供经授权的 PeopleNet 或经过验证的 YOLO nvinfer 配置/模型。
- 当前 `configs/config.yaml` 为本地 face-enabled 验收显式开启人脸与 AdaFace；吸烟/进食/喝水/搬运模型仍全部关闭。任一模块关闭时不创建相应推理组件、不读模型、不占该模型的显存。
- 任一外部模型打开后缺文件会快速报错；模型存在但 Tensor/Parser 不匹配时，DeepStream 会中止并输出具体插件错误。
- 使用现有视频前仍须确认素材授权和代表性；把模型准备好后，才是预期的 `docker compose up` 一键运行状态。

## 架构

```text
file / RTSP (single or multi)
          │
    nvurisrcbin + NVDEC
          │
      nvstreammux
          │
  person nvinfer ── nvtracker (NvDCF/NvSORT/DeepSORT adapter)
          │
          ├── face nvinfer ── multi-frame quality fusion ── AdaFace 512D
          │                                             └── pgvector cosine match
          ├── enabled behavior nvinfer(s)
          │
          └── event/snapshot manager ── unannotated crops + metadata
          │
   tiler + nvdsosd + NVENC ── output/result.mp4
```

关键目录：

```text
configs/       运行、nvinfer、tracker 与模型元数据模板
database/      pgvector 初始化 SQL
docker/        DeepStream 9.0 / PyDS 1.2.3 源码构建镜像
models/        用户挂载的模型（Git 忽略）
scripts/       预检、运行、测试、转换、导入/导出、部署
src/           模块化 Python 业务与 DeepStream 适配层
tests/         单元/配置测试
videos/        测试视频（Git 忽略）
output/        结果、事件与截图（Git 忽略）
```

## Windows 11 + RTX 4090 快速开始

前置条件：较新的 NVIDIA Windows 驱动、WSL2 后端的 Docker Desktop、Docker Compose v2、Docker Desktop 的 WSL/GPU 集成。DeepStream 在 Linux 容器中运行，不直接访问 Windows 摄像头；本地测试统一使用视频文件。

1. 创建本地环境文件并更改开发数据库密码：

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

2. `videos/test.mp4` 可做管线测试；精度测试前请确认素材授权和代表性。将模型放入 `models/`。如使用 PeopleNet，可在已安装并认证 NGC CLI 的 WSL/Linux 环境显式下载：

   ```bash
   scripts/download-models.sh --peoplenet --accept-license
   ```

   脚本不会接收或保存 API key；`--accept-license` 表示操作者已审阅适用条款。下载后仍需核对清单/SHA256，并按 [模型接入](docs/model-integration.md) 指向实际 ONNX、labels、calibration 文件，完成 nvinfer 配置与精度验证。

3. 在 host Python 环境安装工程依赖（建议 venv），然后运行预检：

   ```powershell
   python -m pip install -e .
   powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
   ```

   预检会在 host Python 中执行实际配置/资产验证。默认文件源缺失、人员 nvinfer 配置缺失、既无可用 person engine 也无源模型、labels 等必需资产缺失都会直接失败，不会再以警告后继续。若 host 已安装 `ffprobe`，还会要求所有文件源为 H.264/H.265，并核对 `nominal_fps` 与实际帧率（5% 或至少 0.5 FPS 容差）。使用其他 env/config 可传 `-EnvFile`、`-Config`。

4. 构建并在后台启动识别服务：

   ```powershell
   docker compose up --build -d
   ```

   默认 `APP_DOCKERFILE=docker/Dockerfile` 会从 NGC 拉取较大的 DeepStream 镜像。若已从 NVIDIA 官方渠道取得并审核 `docker/cache/deepstream-9.0_9.0.0-1_amd64.deb`，可在本地 `.env` 设置 `APP_DOCKERFILE=docker/Dockerfile.sdk`：该备用路径从 CUDA 13.0.2 基础镜像安装 CUDA 13.1 libraries/compiler、TensorRT 10.14.1、cuDNN 9.17.1 与 DeepStream 9.0 SDK deb。deb 被 Git 忽略但必须保留在 Docker build context；不得提交或重新分发。两条路径都从 NVIDIA 官方 `deepstream_python_apps` 的固定 commit `8ad0349ed7a496fae35ebb21c350641727070b89` 构建 PyDS 1.2.3，并安装相同 native parsers、PyCUDA、NumPy 1.26.0 和应用依赖。`PGVECTOR_IMAGE` 可选择部署方已审核的 pgvector tag；切换 PostgreSQL major 版本前必须迁移数据，不能直接复用旧 major 的数据目录。

5. 打开 <http://127.0.0.1:8080> 创建任务，并观察服务状态：

   ```powershell
   docker compose ps
   docker compose logs -f app postgres
   ```

   每个任务的状态、日志与结果位于独立目录：

   ```text
   output/tasks/<task_id>/status.json
   output/tasks/<task_id>/pipeline.log
   output/tasks/<task_id>/result.mp4
   output/tasks/<task_id>/events.jsonl
   output/tasks/<task_id>/snapshot/
   ```

## Web 识别服务

Compose 默认运行长期 `serve` 进程，而不是立即处理 `configs/config.yaml` 中的单个视频。控制台支持两种任务输入：

- 本地文件：选择视频后点击“开始分析”。文件以原始请求体上传到 `uploads/<upload_id>/`；服务优先用 `ffprobe` 检查 H.264/H.265、单视频流、帧率和分辨率。精简镜像未提供 `ffprobe` 时会使用 OpenCV 校验编码并实际解码首帧，但该降级路径不能判断容器中是否还有第二路视频流。默认上传上限为 2048 MiB，可用 `.env` 中的 `SERVICE_MAX_UPLOAD_MB` 调整。
- RTSP：切换到“RTSP 视频流”，填写 `rtsp://` 或 `rtsps://` 地址、摄像头 ID 和真实 FPS 后启动。凭据只在本地页面提交；不要把带密码的 URL 写入仓库或日志。RTSP 任务提供实时预览、事件和截图，默认不生成无限增长的单一 MP4；文件任务才生成完成后可下载的 `result.mp4`。

任务页会轮询服务和任务状态，并显示运行任务的 MJPEG 预览。可以停止单个任务；“重启服务”会终止当前活动任务并创建新的服务 generation。文件自然到达 EOS、手动停止或无人超时后，任务详情会给出相应状态；完整 MP4 仅在任务结束并完成封装后可用。

无人超时默认为 10 秒，可通过 `SERVICE_IDLE_TIMEOUT_SEC` 设置，也可在创建任务时单独覆盖。它按**视频流时间**计算：从任务开始或最近一次检测到人形后，连续达到指定时长没有人形即结束任务。Web 服务会按文件时间戳节奏回放上传视频，以便页面持续显示分析过程；超时判定仍以视频时间为准，不受机器瞬时负载影响。

常用接口如下；上传接口接收文件原始字节，不是 `multipart/form-data`：

```text
GET  /api/service
GET  /api/tasks
POST /api/uploads?filename=<name>
POST /api/tasks                    # source_type=file 或 rtsp
POST /api/tasks/<task_id>/stop      # 可用空请求体，或 application/json 的 {}
POST /api/service/restart           # 停止活动任务并切换服务 generation
GET  /api/tasks/<task_id>/stream.mjpg
```

Compose 只把服务发布到宿主机回环地址 `127.0.0.1:${SERVICE_PORT:-8080}`。服务当前没有内置账号、TLS 或跨租户隔离；不要把端口映射改为 `0.0.0.0` 后直接暴露到局域网或互联网。确需远程访问时，应在前面配置带 TLS、认证和访问控制的反向代理。直接运行 CLI 时也应显式使用安全监听地址：

```bash
python -m deepstream_ai serve --config configs/config.yaml --host 127.0.0.1 --port 8080
```

如需原来的单 Pipeline 模式，可显式执行 `python -m deepstream_ai run --config configs/config.yaml`；该模式仍使用配置中的 `output/result.mp4`、`output/events.jsonl` 和 `output/snapshot/`。

## 输入与频率配置

默认单文件输入在 `configs/config.yaml` 的 `source` 节。RTSP 使用：

```yaml
source:
  type: rtsp
  camera_id: gate-01
  url: ${GATE_01_RTSP_URL}
  nominal_fps: 25 # set this to the real camera FPS
  latency_ms: 300
  reconnect_interval_sec: 10
```

凭据放在 `.env`/生产 secret manager，不提交进 YAML。多路输入使用 `sources` 数组；只要数组非空，它就优先于 `source`。每路 `camera_id` 必须唯一。目标推理频率独立配置：

```yaml
inference:
  person_fps: 5
  face_fps: 2
  behavior_fps: 1
```

运行时按源 `nominal_fps` 换算 `nvinfer interval`，不是在 Python 中逐帧取模。完整字段见 [配置说明](docs/configuration.md)。

## 模型与行为开关

行为模型以每模型独立 GIE 接入：

```yaml
behavior:
  smoking:
    enabled: true
    config_file: /workspace/configs/nvinfer/smoking.txt
    model: /workspace/models/smoking.engine
    unique_id: 11
    labels: [smoking]
    threshold: 0.50
```

`.pt` 不能直接交给 `nvinfer`。`scripts/convert-model.sh` 只实现元数据 schema v2 明确声明的 Ultralytics YOLOv8/YOLOv9/YOLO11 detect 合约：动态 batch ONNX、TensorRT `min/opt/max` profile、labels 和 SGIE nvinfer 配置一次生成。默认 profile 为 `1/8/16`，`nvinfer batch-size=16`，脚本会拒绝静态 batch、profile 覆盖不足、类别数/顺序或 parser 合约不一致。INT8 必须提供与实际 profile、预处理和代表性数据匹配的 calibration cache；任意 PyTorch pickle 不会被假装成可自动转换模型。

`.pt` 反序列化可能执行代码，只能在隔离转换环境打开可信权重，禁止把外部上传文件直接交给转换脚本。metadata schema v2 要求预先写入审核过的 `checkpoint.sha256`，脚本在调用 Ultralytics 前逐字节核对。

镜像会编译工程内的动态类别 `libnvdsinfer_custom_yolo_dynamic.so`。它专门支持
Ultralytics YOLO8/9/11 detect 的 `[rows,4+C]` 或 `[4+C,rows]` 原始输出，类别数读取
`num-detected-classes`，不会套用官方 COCO 示例 parser 中硬编码的 80 类。TensorRT 推理仍在
GPU；host parser 只生成 NMS 前候选。其他输出结构必须换匹配 parser，不能把“库可加载”当成
“模型结果正确”。

```bash
# 在已钉死 Ultralytics/ONNX 版本的隔离转换容器内：
bash scripts/convert-model.sh \
  --input models/smoking.pt \
  --metadata configs/models/smoking.metadata.json \
  --onnx models/smoking.onnx \
  --engine models/smoking.engine \
  --labels-file models/smoking.labels.txt \
  --nvinfer-config configs/nvinfer/smoking.txt \
  --precision fp16
```

上述命令须在已另行安装元数据所钉死 Ultralytics/ONNX 版本的隔离转换容器内运行；生产运行镜像不自动安装训练/导出工具。脚本导出后还会核对 ONNX 动态 batch、`4+C` 原始 detect 输出、架构标识与 class names，构建成功后逐文件原子替换产物，最后更新 metadata 作为提交记录。详情见 [模型接入](docs/model-integration.md)。

## PostgreSQL / pgvector

Compose 会启动 `pgvector/pgvector:pg16`，健康检查同时确认 PostgreSQL 可连接且 `vector` 扩展存在。首次建卷时自动创建 `t_worker_face_vector`（`vector(512)`、cosine HNSW）和事件表。连接串由 `DATABASE_DSN` 注入，配置文件不保存密码。

已有卷不会自动重跑初始化 SQL。开发环境确认可丢弃数据后才可执行 `docker compose down -v`；生产必须采用迁移、备份与恢复演练。

## 测试与 VSCode 调试

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test.ps1
```

或在 Linux/WSL：

```bash
scripts/test.sh
```

VSCode 任务提供预检、构建、启动、日志、测试与关闭。调试时选择 `DeepStream: attach container`：预启动任务会以 `debugpy` 在容器内等待，源代码通过 `/workspace/src` 映射。不要同时让普通 `app` 和调试容器写同一个 `output/result.mp4`。

## 镜像导出与 Ubuntu 部署

Windows 导出：

```powershell
scripts/export-image.ps1 `
  -EnvFile .env `
  -Archive output/deepstream-ai-platform.tar
```

Linux/WSL 等价命令：

```bash
scripts/export-image.sh \
  --env-file .env \
  --archive output/deepstream-ai-platform.tar
```

导出脚本先用指定 env file 渲染 Compose，再从 `services.app.image` 读取唯一真实 tag，随后用同一 env file 构建并导出；它没有独立的 image 参数，因此不会在 `APP_IMAGE` 改名后误导出旧的默认 tag。

脚本同时生成 SHA256 文件。将工程配置、所需模型/解析器、视频或 RTSP 配置以及归档安全传到服务器，然后按 [Ubuntu 生产部署](docs/deployment-ubuntu.md) 执行：

```bash
scripts/deploy-ubuntu.sh --archive /srv/deepstream-ai/deepstream-ai-platform.tar
```

该归档只包含业务镜像；目标机需预先拉取或离线导入 `pgvector/pgvector:pg16`。

Compose 是推荐方式，因为它管理 pgvector、健康检查、GPU 预留和挂载。若外部数据库已准备好，也可直接运行镜像：

```bash
docker run --rm --gpus all \
  -e DATABASE_DSN="$DATABASE_DSN" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v "$PWD/configs:/workspace/configs:ro" \
  -v "$PWD/models:/workspace/models:ro" \
  -v "$PWD/videos:/workspace/videos:ro" \
  -v "$PWD/output:/workspace/output" \
  deepstream-ai-platform:dev \
  python3 -m deepstream_ai --config /workspace/configs/config.yaml
```

## 运维与排障

- 运行/停止/日志/数据库备份建议见 [运维手册](docs/operations.md)。
- GPU、解码、nvinfer、TensorRT engine、RTSP、编码与 pgvector 常见问题见 [排障手册](docs/troubleshooting.md)。
- 完整 face-enabled 文件回放结束后，按 [只读运行验收](docs/acceptance-face-run.md) 核对容器、日志、MP4、JSONL events 与 snapshot。
- 生产部署、驱动/Toolkit 与回滚见 [Ubuntu 部署](docs/deployment-ubuntu.md)。

官方兼容性资料：

- [DeepStream 9.0 Docker containers](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_docker_containers.html)
- [DeepStream 9.0 documentation](https://docs.nvidia.com/metropolis/deepstream/9.0/index.html)
- [NVIDIA DeepStream Python Apps releases](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases)

### 为什么没有直接升级到 9.1

当前 Pipeline 通过 PyDS 读取 `NvDsBatchMeta` 和 GPU surface；NVIDIA 已将 Python bindings/sample apps 标记为 deprecated，DeepStream 9.1 的推荐 Python 路线是 Service Maker，且不再承诺现有 PyDS ABI。因而默认固定在已有源码兼容点的 DeepStream 9.0/PyDS 1.2.3，而不是把 9.1 tag 当作无风险替换。迁移到 9.1 需要独立分支把建图、metadata、surface 访问和对象编码能力改到 Service Maker/PyServiceMaker，完成行为一致性与性能回归后再切换；不能只改 Docker tag。
