# 测试视频

当前工作区的 `test.mp4` 已转码为 NVDEC 兼容的 H.264；原 MPEG-4 Part 2 文件保留为 `test-original-mpeg4.mp4`。使用前须确认素材授权；检测、跟踪、人脸、身份、行为和截图验收还须确认它具有代表性。媒体文件不提交到 Git，因此全新 clone 不包含 `test.mp4`，预检会按设计失败，直到使用方提供获授权的 H.264/H.265 文件并让 `nominal_fps` 与真实 FPS 一致。

建议素材：固定帧率、可由 NVIDIA 硬件解码器读取、包含多人/遮挡/目标行为，并另行保留标注用于回归测试。可先验证容器内解码：

```bash
docker compose run --rm --no-deps app \
  gst-launch-1.0 filesrc location=/workspace/videos/test.mp4 \
  ! qtdemux ! h264parse ! nvv4l2decoder ! fakesink
```
