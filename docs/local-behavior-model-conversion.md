# 本地行为模型 DeepStream 转换

本项目不把大模型文件提交到 GitHub。宿主机 `./models` 会挂载为容器内 `/workspace/models`，模型转换与正式识别使用同一套 DeepStream/TensorRT 基础环境。

## 1. 当前约定的本地输入

把以下文件放在项目根目录 `models/`：

```text
models/
├── yolo11n.onnx   # 吃东西 + 喝水，共享两分类检测模型
├── smoking.pt     # 吸烟检测模型
├── phone.pt       # 打电话检测模型
└── fire.onnx      # 火焰检测模型
```

现有人形、NvDCF、SCRFD、AdaFace 模型不参与本次转换，也不会被覆盖。

## 2. 一次性转换

在有 NVIDIA GPU、Docker、NVIDIA Container Toolkit 的目标机器上执行：

```bash
docker compose --profile tools build model-converter
docker compose --profile tools run --rm model-converter
```

转换工具会：

1. 检查输入文件；
2. `smoking.pt`、`phone.pt` 用 Ultralytics 导出 raw detect ONNX；
3. 校验所有 ONNX 是否符合本项目 `NvDsInferParseCustomYoloDynamic` 的 raw YOLO detect 输出契约；
4. 使用当前容器 TensorRT `trtexec` 构建 FP16 Engine；
5. 从模型 metadata/checkpoint 读取真实类别名称；
6. 生成 labels 文件；
7. 生成 DeepStream nvinfer 配置；
8. 写出 SHA256、tensor shape、类别与生成文件路径到 manifest；
9. 再次校验业务类别与类别顺序，错误时以非 0 状态退出。

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
├── phone.pt
├── phone.onnx
├── phone.engine
├── phone.labels.txt
├── fire.onnx
├── fire.engine
├── fire.labels.txt
└── deepstream-local-models.manifest.json

configs/nvinfer/
├── eat-drink.txt
├── smoking.txt
├── phone.txt
└── fire.txt
```

这些 `.onnx/.pt/.engine` 文件继续被 `.gitignore` 排除，只保留在部署机器。

## 4. 类别契约

当前生产配置明确要求：

```text
yolo11n.onnx : class 0 = eating, class 1 = drinking
smoking.pt    : smoking
phone.pt      : phone
fire.onnx     : fire
```

转换器会把常见训练标签规范化，例如：

```text
eat -> eating
drink -> drinking
smoke/cigarette -> smoking
calling/cell phone/mobile phone -> phone
flame -> fire
```

如果 `yolo11n.onnx` 实际类别顺序是 `[drinking, eating]`，转换命令会明确失败，不允许静默把喝水和吃东西颠倒。正式服务的 capabilities 检查也会比较生成 labels 与 `config.yaml`，避免绕过转换工具后加载错误类别。

## 5. 生产运行关系

### 吃东西 / 喝水

`yolo11n.engine` 只加载一次，是一个共享 SGIE：

```text
PeopleNet person -> NvDCF -> yolo11n.engine
                             ├─ eating   -> EatingProcessor
                             └─ drinking -> DrinkingProcessor
```

REST 仍然保持两个独立开关。只选喝水时，共享模型执行，但吃东西结果不会进入业务 Processor；只选吃东西时同理。

### 吸烟

```text
PeopleNet person -> NvDCF -> smoking.engine -> SmokingProcessor
```

### 打电话

```text
PeopleNet person -> NvDCF -> phone.engine -> Phone BehaviorScenarioProcessor
```

测试 Web 页面提供独立“打电话”开关。

### 火焰

火焰不应该只在 person bbox 内识别，因此 `fire.onnx` 被转换为 **full-frame** DeepStream 配置：

```text
configs/nvinfer/fire.txt
models/fire.engine
```

本次只完成正确转换与配置，不把它错误接成 per-person SGIE。测试页面会显示“火焰（已转换，待接入）”且不可勾选。后续接入时应增加独立的全画面 Fire Processor / inference branch，而不是修改人形、人脸主链路。

## 6. 精度参数

生成的行为 nvinfer 配置初始检测阈值为 `0.35`，业务层仍使用 `config.yaml` 中每个行为的 `threshold`（当前为 `0.50`）做最终过滤。先用实际摄像头视频统计置信度分布，再单独调整相应行为阈值；不要同时修改 PeopleNet、NvDCF、SCRFD 或 AdaFace 参数。

## 7. 单模型转换

需要排查单个模型时，可以直接运行转换器并使用 `--only`：

```bash
docker compose --profile tools run --rm --entrypoint python3 model-converter \
  /workspace/scripts/prepare-local-behavior-models.py \
  --model-root /workspace/models \
  --config-root /workspace/configs/nvinfer \
  --device 0 \
  --only smoking \
  --force
```

支持：`eat_drink`、`smoking`、`phone`、`fire`。

## 8. 失败原则

以下情况不会生成“假成功”配置：

- `.pt` 不是 Ultralytics detect checkpoint；
- ONNX 不是单输入 raw YOLO detection 输出；
- 模型带 embedded NMS/end-to-end output，与当前 parser 不兼容；
- tensor rank/shape 与 raw YOLO parser 契约不符；
- 类别数或类别顺序与当前业务配置不一致；
- `trtexec` / TensorRT 构建失败；
- 生成 Engine 为空。

遇到这些问题应保留原模型，针对该模型的实际输出契约调整转换方式，不要为了启动成功而关闭检查。
