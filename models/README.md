# 模型目录

版本库不跟踪或分发权重与 TensorRT engine；只随工程提供无权重的类别标签。本地验收生成的忽略文件可能出现在该目录，仍须确认模型许可后再用于部署：

```text
models/
  person.onnx
  person.engine
  person.labels.txt
  tracker/
    resnet50_market1501_aicity156.onnx
    resnet50_market1501_aicity156.onnx_b16_gpu0_fp16.engine
  face-deepstream.onnx
  face.engine
  face.labels.txt
  adaface_ir50.onnx / adaface.engine
  smoking.pt / smoking.onnx / smoking.engine
  eating.pt  / eating.onnx  / eating.engine
  drinking.pt / drinking.onnx / drinking.engine
  carrying.pt / carrying.onnx / carrying.engine
```

YOLO11 与标准 9-output SCRFD parser 已从 `native/` 编译进镜像，不需要放进本目录；模型权重
和 engine 仍由使用方提供。`face.engine` 必须严格符合 `native/scrfd_parser/README.md` 的
stride/anchor/9-output 合约，`face.labels.txt` 内容为一行 `face`。

TensorRT engine 与 GPU 架构、TensorRT/CUDA 版本、精度和输入 shape 绑定。应在最终部署镜像/同构 GPU 上使用 `scripts/convert-model.sh` 构建，不能把开发机生成的 engine 当作通用文件。

任一模型只有在 `configs/config.yaml` 中显式 `enabled: true` 才会加载；关闭时即使文件存在也不读文件、不创建推理组件、不占该模型的 GPU 显存。启用前必须同时确认预处理、类别顺序、输出 tensor 和 DeepStream custom parser 与训练导出一致。

仓库仅为已审核的 PeopleNet 和 NvDCF ReID 坐标提供显式下载入口：先在主机安装并认证 NVIDIA NGC CLI，阅读并接受适用条款，再执行：

```bash
scripts/download-models.sh --peoplenet --accept-license
scripts/download-models.sh --reid --accept-license
```

ReID 使用 NGC `nvidia/tao/reidentificationnet:deployable_v1.2` 中的 `resnet50_market1501_aicity156.onnx`。下载脚本只接受精确文件名、`96398132` 字节和 SHA256 `0e21d09278508ec835955f422a9fdd3cd59b2a6ecdef98d705f388f33cebac2b`；它在 `models/tracker/` 内暂存，校验后原子安装。已有文件正确时幂等成功，已有文件不符时拒绝覆盖。脚本不接收、打印或保存 API key。SCRFD/RetinaFace、AdaFace 与行为权重仍不自动下载，避免把未授权或错误模型悄悄用于生产。

在目标部署 GPU 和目标 DeepStream/TensorRT 容器中构建动态 batch `1/8/16` FP16 engine：

```bash
docker compose run --rm --no-deps \
  --entrypoint /usr/src/tensorrt/bin/trtexec app \
  --onnx=/workspace/models/tracker/resnet50_market1501_aicity156.onnx \
  --saveEngine=/workspace/models/tracker/resnet50_market1501_aicity156.onnx_b16_gpu0_fp16.engine \
  --minShapes=input:1x3x256x128 \
  --optShapes=input:8x3x256x128 \
  --maxShapes=input:16x3x256x128 \
  --fp16
```

TensorRT engine 不能跨 GPU 架构、TensorRT/CUDA 运行时或平台复制复用，必须在最终目标环境重建。启用 ReID 的 tracker 配置所引用的 ONNX/engine 任一缺失或校验不符时，部署必须 fail-fast，禁止静默退回无 ReID tracker。

人员模型的默认审核坐标是 NGC `nvidia/tao/peoplenet:deployable_quantized_onnx_v2.6.3`。该坐标不等于再分发许可：由使用方登录 NGC、接受适用条款、记录下载文件 SHA256，并按其输入/输出和 INT8 calibration 要求配置。NVIDIA 的最新 DeepStream-Yolo `v9.1.0` 面向 DeepStream 9.1，不能直接拿来给本工程的 DeepStream 9.0 runtime；只有找到/构建并验证与 9.0 匹配的 parser release 后才能启用。
