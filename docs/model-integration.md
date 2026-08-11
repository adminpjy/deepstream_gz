# 模型接入与转换

## 原则

TensorRT engine 不是通用模型文件。它与 TensorRT/CUDA 版本、GPU compute capability、精度、batch/profile、插件和输出 parser 绑定。应在与生产一致的 DeepStream 镜像和同代 GPU 上构建并记录：来源/许可、训练代码版本、checkpoint SHA256、预处理、输入输出 tensor、类别顺序、opset、TensorRT 版本、GPU 型号、精度、校准集版本和验收指标。

PyTorch `.pt` 通常包含 pickle 反序列化面，可能执行代码。只在隔离、无生产凭据的转换环境中打开来源可信且 SHA256 已核对的 checkpoint；不要让上传文件直接触发本脚本。

仓库只为审核坐标明确的 PeopleNet 提供自动获取。先在主机安装 NVIDIA NGC CLI、由操作者交互完成 `ngc config set` 并审阅适用条款，然后运行：

```bash
scripts/download-models.sh --peoplenet --accept-license \
  --destination models/peoplenet
```

脚本执行官方 `ngc registry model download-version`，不接收、打印或保存 API key。NGC 可能生成带版本名的子目录；下载后需核对 manifest/SHA256，再把 nvinfer 路径指向实际文件。其他模型不会猜测下载地址或替使用方接受第三方许可，获取必须由部署组织审批并留痕。

## 人员检测

优先选择获得授权的 NVIDIA PeopleNet，或者经过现场数据验证的 YOLO 人员检测模型。PeopleNet 的默认审核坐标为 NGC `nvidia/tao/peoplenet:deployable_quantized_onnx_v2.6.3`；使用方仍须登录、接受条款并保存文件版本/SHA256。完成后：

1. 将 engine/ONNX、labels 和必要的 custom parser 放到 `models/`。
2. 从 `configs/nvinfer/person.example.txt` 建立实际配置（可直接改名为 `person.txt`）。
3. 校对输入尺寸、色彩顺序、缩放、类别 ID、阈值、聚类/NMS、batch 和 parser。
4. 将 `configs/config.yaml` 的 `person.config_file` 指向实际文件。
5. 以多路 batch、遮挡、小目标和夜间录像检查检测召回率与 Tracker ID switch。

PeopleNet/YOLO 输出结构不同；不能只替换 engine 路径而保留另一架构的 parser。

NVIDIA DeepStream 示例 YOLO parser 面向其固定示例输出，其中 YOLO8/11 路径按 COCO 80 类
读取，不能安全复用于单类行为模型。本工程因此编译 `native/yolo_parser` 的动态类别 host parser，
安装到 `/opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo_dynamic.so`。它只接受
Ultralytics YOLO8/9/11 detect 的 `[rows,4+C]` 或 `[4+C,rows]` 原始输出；TensorRT 推理仍在
GPU，nvinfer 使用 `cluster-mode=2` 执行 NMS。带 objectness、end-to-end NMS、分割、姿态或 OBB
的模型必须换专用 parser。

## SCRFD / RetinaFace 与 AdaFace

镜像内置 `native/scrfd_parser`，支持一个明确、可测试的部署 ABI：SCRFD 五点模型、stride
8/16/32、每点 2 anchors、每个 stride 分别输出 score `[N,1]`、bbox distance `[N,4]`
和 landmark `[N,10]`，总计 9 个 FP32 host tensors。行主序和通道主序均可；6/10/15
输出或无关键点/anchor-free 变体会快速报错，不能靠猜测接入。

原生 parser 使用同一公式产生 face child bbox，`output-tensor-meta=1` 同时把原始输出挂到
parent person 的 `obj_user_meta_list`。Python 按 `unique_id` 读取、在 pad probe 返回前复制张量，
逆变换 symmetric letterbox/person ROI，再把 proposal 与 face child 按 IoU 一对一匹配，得到
左眼、右眼、鼻尖、左嘴角、右嘴角。`landmark_threshold` 必须与 nvinfer
`pre-cluster-threshold` 一致；缺失、歧义或 IoU 不足均 fail closed。AdaFace 默认
`require_landmarks: true`，不会悄悄退化成 bbox 拉伸。

将兼容的 `face.engine` 和单行 `face.labels.txt` 放入 `models/` 后再启用 `face`。RetinaFace
或其他 SCRFD 导出 ABI 必须提供自己的 bbox/tensor decoder 并新增回归测试，不能复用本 parser
的名称假装兼容。AdaFace engine 也必须由使用方按其模型许可提供，并确认输入为对齐后的
112x112 人脸、输出维度 512。

InsightFace buffalo_l 的 `det_10g.onnx` 一类导出可能把 9 个输出写成 `[rows,width]`，而
DeepStream/TensorRT 的显式 batch parser 需要 `[1,rows,width]`。确认来源与权重许可后，可在
隔离的转换环境运行：

```bash
python scripts/prepare-scrfd-onnx.py models/face.onnx models/face-deepstream.onnx
```

脚本只接受固定 640、9 个 FP32 输出、stride 8/16/32、2 anchors 的严格 SCRFD 合约，保留原
binding 名并为每个输出增加 batch 轴；默认拒绝覆盖目标。转换后必须在目标 DeepStream/GPU
重新生成 `face.engine`，旧 engine 不能复用。该结构变换不会解决权重来源或商业授权问题。

metadata probe 会在 GPU surface 生命周期结束前复制一份完整 RGBA 帧，并交给有界异步队列；这是截图/裁剪正确性的安全基线，不是零拷贝方案。高分辨率多路部署应监控队列丢帧和内存带宽，并在压测后再引入 ROI 或 GPU-side crop，不能让 Python worker 持有已经回收的 surface 指针。

AdaFace 输入由五点 Umeyama similarity transform 对齐为 `112×112`，输出规范化 512 维 embedding。实际模型的颜色顺序、均值/方差、对齐模板和 tensor 名必须以导出记录为准。用相同身份/不同身份对构建 ROC/DET，按现场误识率目标标定阈值；禁止用单帧结果直接下结论。

## 行为 YOLO `.pt → ONNX → TensorRT`

本工程只有一条内置行为模型合约：Ultralytics YOLOv8、YOLOv9 或 YOLO11 的 `detect` 模型，单个原始输出为 `[N,4+C,rows]` 或 `[N,rows,4+C]`，后续由工程 parser 生成候选、由 `nvinfer cluster-mode=2` 做 NMS。分类、分割、姿态、OBB、含 objectness 的旧 YOLO 输出和 end-to-end NMS 均不属于该合约。

先复制 schema v2 元数据模板：

```bash
cp configs/models/behavior.metadata.example.json \
   configs/models/smoking.metadata.json
```

必须替换所有占位值，并保持这些不变量：

- 输入必须为 `.pt`；metadata 中预先审核的 `checkpoint.sha256` 必须与文件逐字节匹配，校验在反序列化之前完成；
- `architecture` 只能是规范值 `yolov8`、`yolov9` 或 `yolo11`，`task=detect`；
- `exporter.name=ultralytics`，`exporter.version` 必须是转换环境中实际安装的精确审核版本；
- `input.shape` 为 `[-1,3,H,W]`、`dynamic_batch=true`，profile 的 C/H/W 与输入一致；
- profile 满足 `min <= opt <= max`，`max` 至少为 16，且覆盖 `deepstream.batch_size`；默认 `min/opt/max=1/8/16`、SGIE batch 为 16；
- `labels` 非空、有序且唯一，`deepstream.num_detected_classes == len(labels)`；
- 人员 GIE ID 为 1，行为模型 `process_mode=2` 且只处理 person class 0；
- parser contract、动态 parser 库、函数名和 `cluster_mode=2` 必须与模板完全一致。

`configs/nvinfer/behavior-yolo11.example.txt` 只是便于审核的参考输出，不应手工复制为生产配置。脚本从 metadata 自动生成 labels 与对应 SGIE 配置，因而 engine profile、`batch-size`、类别数、路径、阈值和 parser 不会各自漂移。完成后仍须让 `configs/config.yaml` 中 behavior 的 `model`、`config_file`、`unique_id`、labels 和 threshold 与该 metadata 对齐；应用启动预检会检查 YAML model 与 nvinfer engine 路径。

先做不加载 checkpoint、不要求 `yolo/trtexec` 的结构验证：

```bash
bash scripts/convert-model.sh \
  --input models/smoking.pt \
  --metadata configs/models/smoking.metadata.json \
  --onnx models/smoking.onnx \
  --engine models/smoking.engine \
  --labels-file models/smoking.labels.txt \
  --nvinfer-config configs/nvinfer/smoking.txt \
  --validate-only
```

实际转换环境必须有 metadata 钉死版本的 `yolo`/Python `ultralytics` 与 `onnx`，以及目标 DeepStream/TensorRT 镜像中的 `trtexec`。生产运行镜像刻意不携带训练/导出依赖；应使用隔离、无生产凭据的临时转换镜像，并与部署 GPU、TensorRT、CUDA 和 parser 版本一致：

```bash
bash scripts/convert-model.sh \
  --input models/smoking.pt \
  --metadata configs/models/smoking.metadata.json \
  --onnx models/smoking.onnx \
  --engine models/smoking.engine \
  --labels-file models/smoking.labels.txt \
  --nvinfer-config configs/nvinfer/smoking.txt \
  --precision fp16
```

脚本传给 Ultralytics `dynamic=True,nms=False`，然后验证 ONNX graph 确实是动态 batch、唯一 rank-3 原始 detect 输出中含 `4+C` 维，并核对 ONNX metadata 的 `task`、架构标识和有序 class names。TensorRT 使用显式 `--minShapes/--optShapes/--maxShapes`；不会用固定 `--shapes=1x...` 构建一个却让 nvinfer 以 batch 16 运行。

INT8 示例：

```bash
bash scripts/convert-model.sh \
  --input models/smoking.pt \
  --metadata configs/models/smoking.metadata.json \
  --onnx models/smoking.onnx \
  --engine models/smoking-int8.engine \
  --labels-file models/smoking.labels.txt \
  --nvinfer-config configs/nvinfer/smoking-int8.txt \
  --precision int8 \
  --calibration-cache models/smoking.calib
```

脚本不会从图片自动伪造校准 cache。cache 必须由与部署 profile/预处理完全一致的 TensorRT calibrator/TAO 工作流和代表性现场样本生成，并在 metadata 中记录 `dataset_id` 与 `preprocessing_id`。

所有昂贵步骤和语义校验通过后，脚本才把 ONNX、engine、labels、nvinfer config 分别写入同目录临时文件并用 `os.replace` 原子替换；metadata 最后更新，记录输入和产物 SHA256、精度、profile、nvinfer batch、parser contract 及 exporter 版本。多个文件无法获得跨文件系统事务语义；若提交阶段遇到磁盘故障，metadata 不会提前宣告成功，清理已有产物后重跑即可。已存在产物默认拒绝覆盖，人工核验目标后才能使用 `--force`。

## 上线门禁

转换后至少完成：

1. 训练框架与 ONNX 同输入输出数值对比。
2. ONNX 与 TensorRT bbox/class/confidence 对比。
3. custom parser 后的 DeepStream 结果与参考 NMS 对比。
4. FP16/INT8 精度回归和吞吐/显存测试。
5. 错误场景、多人遮挡、跨摄像机和长时间稳定性测试。
6. engine、parser、metadata 与镜像生成不可变版本号及 SHA256。

只有全部通过后才把相应 `enabled` 改为 `true`。
