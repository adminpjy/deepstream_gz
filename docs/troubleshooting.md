# 排障手册

按“主机 GPU → Docker GPU → DeepStream 插件 → 输入解码 → 模型/Parser → 输出编码 → 数据库/业务”的顺序定位，先保留完整日志和版本，不要通过关闭错误检查掩盖根因。

## 容器看不到 GPU

症状包括 `could not select device driver`、`libcuda.so` 缺失、`nvidia-smi` 失败。

```bash
nvidia-smi
docker info
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

Windows 检查 WSL2 后端、支持 WSL 的 NVIDIA 驱动和 Docker Desktop WSL integration；Ubuntu 检查 NVIDIA Container Toolkit 配置并重启 Docker。不要在容器里装内核驱动。

## NGC 镜像拉取失败

确认能访问 `nvcr.io`、磁盘空间和代理。某些网络/限流场景需要按 NGC 官方流程 `docker login nvcr.io`。不要把 NGC API key 写进 Dockerfile、Compose 或终端截图。

## `pyds` 导入失败

```bash
docker compose run --rm --no-deps app python3 -c \
  "import sys,pyds; print(sys.version); print(pyds.__file__)"
```

工程默认 DeepStream 9.0 + Ubuntu 24.04 + Python 3.12 + PyDS 1.2.3。PyDS 从 NVIDIA 上游固定 commit 源码构建，NumPy 精确固定为 1.26.0；若改了 `DEEPSTREAM_IMAGE`，必须重新验证/构建对应 PyDS，不要从 PyPI 安装同名但无关的包。DeepStream 9.1 已转向 Service Maker，不保证这个 PyDS 组合可直接迁移。

## 视频文件不存在或无法解码

```bash
docker compose run --rm --no-deps app ls -l /workspace/videos/test.mp4
docker compose run --rm --no-deps app \
  gst-launch-1.0 filesrc location=/workspace/videos/test.mp4 \
  ! qtdemux ! h264parse ! nvv4l2decoder ! fakesink
```

确认 Windows 文件已共享给 Docker、路径大小写正确、封装/编码受支持。不要改成 Python/OpenCV 循环绕过硬件 Pipeline。

## RTSP 黑屏、延迟或频繁重连

先从容器内验证网络/DNS，再检查 URL、认证、TCP/UDP、防火墙、摄像机连接上限、时间戳和关键帧间隔。适度调高 `latency_ms`，配置真实 `nominal_fps`。多路相机时逐路加入，按 `camera_id` 查看日志；避免在日志中输出含密码的完整 URL。

## `nvinfer` 无法加载 engine

常见原因：engine 在不同 GPU/TensorRT 版本生成、batch/profile 不含当前路数、插件/parser 缺失、tensor 名/shape 不一致、文件权限或路径错误。

```bash
docker compose run --rm --no-deps app trtexec --loadEngine=/workspace/models/person.engine --dumpLayerInfo
docker compose run --rm --no-deps app \
  ldd /opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo_dynamic.so
```

必要时在目标镜像/目标 GPU 上从 ONNX 重建。不要把 `strict_assets` 设为 false 后继续生产；该选项只适合诊断非模型链路。

## YOLO 有框但类别/坐标错误

检查 RGB/BGR、归一化、letterbox、输入 shape、输出 tensor、类别顺序、NMS、坐标尺度和 parser 版本。用同一张图逐级比对训练框架、ONNX Runtime、TensorRT 和 DeepStream parser 输出。转换成功不等于业务正确。

## Tracker ID 跳变

先确认人员检测召回、bbox 稳定性和 `person_fps`；Tracker 无法弥补长时间漏检。当前 PeopleNet v2.6.3 必须保留 `output_cov`→`output_bbox` 的绑定顺序以及经过回归验证的 Hybrid（DBSCAN+NMS）参数，不能只降低单个 NMS 阈值：现场空场机柜曾在这种配置下形成高置信度假人框。再基于现场遮挡/密度调 `probationAge`、`maxShadowTrackingAge`、IoU/视觉相似度和 tracker 分辨率。保存有标注的长序列衡量 IDF1/ID switch，而不是只看短片主观效果。

## 人脸一直 unknown 或误识

先确认 `face.engine` 是文档约定的 9-output、stride 8/16/32、2-anchor SCRFD 五点模型；其他 SCRFD/RetinaFace ABI 会被严格拒绝。检查日志是否出现 tensor meta/stride/IoU 匹配失败，确认 `landmark_threshold` 与 nvinfer 阈值相同。随后检查五点对齐、AdaFace 预处理、embedding L2 norm、512 维存储和 cosine 距离方向。确认人员库使用同一模型版本生成；不同模型 embedding 不可混用。按现场 ROC 重新标定阈值，并检查多帧候选是否被低质量/侧脸占用。

## 结果 MP4 不可播放

若容器被强杀，`qtmux` 可能未写完索引。使用 `docker compose stop -t 30 app`，让 Pipeline 从 PLAYING 到 NULL。检查 NVIDIA encoder 可用：

```bash
docker compose run --rm --no-deps app gst-inspect-1.0 nvv4l2h264enc
```

同时确认 `NVIDIA_DRIVER_CAPABILITIES` 含 `video`、输出目录可写且磁盘有空间。

## pgvector 不健康或匹配慢

```bash
docker compose exec postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"
docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

初始化 SQL 只在空卷首次运行。旧卷缺扩展时用受控迁移修复。匹配慢时先看查询计划、向量数量、HNSW 索引、连接池与数据库资源；不要直接降低阈值来掩盖查询/embedding 问题。
