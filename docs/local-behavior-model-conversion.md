# 本地行为模型 DeepStream 转换

本项目不把大模型文件提交到 GitHub。宿主机 `./models` 会挂载为容器内 `/workspace/models`，模型转换与正式识别使用同一套 DeepStream/TensorRT 基础环境。

## 1. 当前约定的本地输入

当前生产行为识别只使用以下两份模型：

```text
models/
├── yolo11n.onnx   # 标准 COCO 检测器，复用于吃东西/喝水道具证据
└── smoking.pt     # 独立吸烟检测模型
```

`phone.pt` 已从当前工程的生产识别能力中移除，不再转换、不再加载、也不再提供 Session 开关。

`fire.onnx` 是独立的全画面 3 类分类模型，输出为 `[B,3]`，不属于 raw YOLO detector。它保留在本地用于后续火焰/烟雾独立接入，但不会再阻塞当前吃喝/吸烟模型转换。

现有人形、NvDCF、SCRFD、AdaFace 模型不参与本次转换，也不会被覆盖。

## 2. 一次性转换

在有 NVIDIA GPU、Docker、NVIDIA Container Toolkit 的目标机器上执行：

```bash
docker compose --profile tools build model-converter
docker compose --profile tools run --rm model-converter
```

默认转换只处理：

```text
eat_drink
smoking
```

转换工具会：

1. 检查 `yolo11n.onnx` 与 `smoking.pt`；
2. 将 `smoking.pt` 用 Ultralytics 导出 raw detect ONNX；
3. 校验 ONNX raw YOLO detect 输出契约；
4. 使用当前容器 TensorRT `trtexec` 构建 FP16 Engine；
5. 从模型 metadata/checkpoint 读取真实类别名称；
6. 生成 labels 文件；
7. 生成 DeepStream nvinfer 配置；
8. 写出 SHA256、tensor shape、类别与生成文件路径到 manifest；
9. 校验吃喝 COCO 源类别、业务类别、parser 与 0.45 阈值；
10. 校验吸烟模型类别为 `smoking`，错误时以非 0 状态退出。

## 3. 生成结果

成功后应得到：

```text
models/
├── yolo11n.onnx
├── yolo11n.engine
├── yolo11n.labels.txt
├── smoking.pt
├── smoking.onnx
├── smoking.engine
├── smoking.labels.txt
└── deepstream-local-models.manifest.json

configs/nvinfer/
├── eat-drink.txt
└── smoking.txt
```

这些 `.onnx/.pt/.engine` 文件继续被 `.gitignore` 排除，只保留在部署机器。

## 4. 吃东西 / 喝水规则

`yolo11n.onnx` **不是 eating/drinking 两分类模型**，而是标准 COCO 80 类检测器。当前工程只复用其中与吃喝相关的道具类别，并在 PeopleNet 已经得到的人员 ROI 内运行。

业务判定严格对齐 `opsvision/opvision/eating_drinking.py`：

- 道具置信度必须 `>= 0.45`；
- 道具中心点必须位于人员框顶部 `40%` 区域；
- 喝水道具：`bottle`、`cup`、`wine glass`、`bowl`；
- 吃东西道具：`apple`、`banana`、`sandwich`、`orange`、`pizza`、`donut`、`cake`、`hot dog`。

DeepStream parser 对外只暴露两个业务类别：

```text
class 0 = eating
class 1 = drinking
```

因此当前链路是：

```text
PeopleNet person
      ↓
NvDCF
      ↓
yolo11n.engine（人员 ROI 内 COCO 道具检测）
      ↓
顶部40% + 0.45 + opsvision类别集合
      ├─ eating
      └─ drinking
```

REST 仍保持“吃东西”和“喝水”两个独立开关。任意一个开关打开时，共享 YOLO SGIE 才对该摄像头执行；业务 Processor 只接收该 Session 实际启用的事件类型。

## 5. 吸烟

吸烟不复用 `opsvision` 中的 `EatingDrinking` 规则。旧 `opsvision` 虽然存在 `SMOKING_MODEL` 配置，但实际主逻辑没有加载该模型，`EatingDrinking.infer()` 也不会返回 `smoking`。

当前工程使用独立吸烟模型：

```text
smoking.pt
   ↓ Ultralytics export
smoking.onnx
   ↓ TensorRT
smoking.engine
   ↓ DeepStream person SGIE
SMOKING event
```

这条实现保持不变。

## 6. 行为采样频率

生产配置中行为模型采样频率统一为：

```text
behavior_fps = 5.0
```

所有自适应负载档位也保持 5 FPS，避免短时间吃喝/吸烟动作因为 1 FPS 采样被跳过。该调整只作用于行为 SGIE，不修改 PeopleNet、NvDCF、SCRFD 或 AdaFace 参数。

## 7. 单模型转换

需要排查单个模型时，可以直接运行动态转换入口并使用 `--only`：

```bash
docker compose --profile tools run --rm --entrypoint python3 model-converter \
  /workspace/scripts/prepare-local-behavior-models-dynamic.py \
  --model-root /workspace/models \
  --config-root /workspace/configs/nvinfer \
  --device 0 \
  --only smoking \
  --force
```

当前生产行为链建议只使用：

```text
eat_drink
smoking
```

## 8. 火焰模型说明

`fire.onnx` 已确认是全画面分类模型：

```text
input:  pixel_values [B,C,H,W]
output: logits [B,3]
```

它不能使用 YOLO bbox parser，也不能接在 person SGIE 后面。当前 capabilities 会将 fire 标记为待接入。后续只有在明确 3 个类别顺序和输入预处理契约后，才单独接入 full-frame classifier 分支。

## 9. 失败原则

以下情况不会生成“假成功”配置：

- `smoking.pt` 不是 Ultralytics detect checkpoint；
- `yolo11n.onnx` 不是单输入 raw YOLO detection 输出；
- 模型带 embedded NMS/end-to-end output，与当前 parser 不兼容；
- tensor rank/shape 与 raw YOLO parser 契约不符；
- `yolo11n.onnx` 缺少 opsvision 吃喝规则所需的 COCO 类别；
- 吃喝 parser 不是 `NvDsInferParseCustomYoloEatDrinkCoco`；
- 吃喝 DeepStream 阈值不是 `0.45`；
- 吸烟模型类别不是 `smoking`；
- `trtexec` / TensorRT 构建失败；
- 生成 Engine 为空。

遇到这些问题应保留原模型，针对实际输出契约调整转换方式，不要为了启动成功而关闭检查。
