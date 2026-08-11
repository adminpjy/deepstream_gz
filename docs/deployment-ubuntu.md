# Ubuntu GPU 服务器部署

## 1. 主机准备

推荐使用 NVIDIA DeepStream 9.0 官方支持的 Ubuntu/驱动组合。服务器必须安装 Docker Engine、Docker Compose v2、与 DeepStream/CUDA 兼容的 NVIDIA 驱动及 NVIDIA Container Toolkit。不要在容器内安装主机驱动。运维预检还需要 host Python 3 与本工程依赖，用于在分配 GPU 前运行同一配置/资产验证器；建议在受控 venv 中执行 `python3 -m pip install -e .`。

安装完成后先验证：

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

驱动最低版本以所选运行栈的官方兼容矩阵为准。默认 `APP_DOCKERFILE=docker/Dockerfile` 使用 `nvcr.io/nvidia/deepstream:9.0-triton-multiarch`；受控离线环境也可使用 `docker/Dockerfile.sdk` 和已审核的本地 DeepStream 9.0 SDK deb。备用路径固定 CUDA 13.1、TensorRT 10.14.1 和 cuDNN 9.17.1，deb 仅作为本地 build-context 输入，不得提交或重新分发。升级任一路径时必须同步检查驱动、Python、PyDS、CUDA、TensorRT、cuDNN、GStreamer 插件和已构建 engine，不能只改 tag。当前 PyDS 1.2.3 来自固定 NVIDIA 上游 commit 的源码构建，PyCUDA 也在构建阶段编译。

## 2. 目录与权限

```bash
sudo install -d -m 0750 /srv/deepstream-ai
sudo chown "$USER":"$USER" /srv/deepstream-ai
cd /srv/deepstream-ai
```

将以下内容传入服务器：工程配置和脚本、镜像 tar+SHA256、获得授权的模型/parser、测试视频（如需）。禁止传入开发 `.env` 或未审批的摄像机密码。

`scripts/export-image.*` 只导出业务镜像。它以 `--env-file`（PowerShell 为 `-EnvFile`）渲染 Compose，并从 `services.app.image` 解析实际 `APP_IMAGE`/tag；不要另行手写 tag。目标机还需具有 `PGVECTOR_IMAGE` 指定的数据库镜像；隔离网络环境应在联网中转机对同一精确 tag 执行 `docker save`，传输校验后在目标机 `docker load`，不要临时换成未审计的数据库镜像。PostgreSQL major 升级必须使用受支持的迁移流程，不能把旧 major 数据卷直接挂给新 major。

```bash
scripts/export-image.sh \
  --env-file .env.production \
  --archive output/deepstream-ai-platform.tar
```

```text
/srv/deepstream-ai/
  docker-compose.yml
  pyproject.toml
  src/
  .env
  configs/
  database/
  scripts/
  models/
  videos/
  output/
  deepstream-ai-platform.tar[.sha256]
```

`output/` 与 `models/` 应仅允许运行账号写/读；RTSP 和数据库密码通过受控 `.env`（`chmod 0600`）、Docker secrets 或组织 secret manager 注入。

## 3. 环境和模型确认

```bash
cp .env.example .env
chmod 0600 .env
editor .env
```

必须修改 `POSTGRES_PASSWORD` 和 `DATABASE_DSN`，二者密码保持一致。生产 RTSP URL 建议以环境变量引用。确认 engine 是在目标 TensorRT/同构 GPU 上生成，并核对 parser/labels/metadata SHA256。

运行只读 Compose 校验：

```bash
docker compose config --quiet
```

注意 `docker compose config` 的完整输出会展开 secret，不要把它粘贴进工单或日志。

## 4. 导入并启动

```bash
scripts/deploy-ubuntu.sh \
  --archive /srv/deepstream-ai/deepstream-ai-platform.tar
```

脚本验证归档 checksum、加载镜像、执行 GPU/Compose 预检，然后使用 `--no-build` 启动。观察：

```bash
docker compose ps
docker compose logs -f --tail=200 app postgres
docker inspect --format '{{json .State.Health}}' "$(docker compose ps -q app)"
```

文件源到 EOS 后进程正常结束；RTSP 服务应持续运行。将 `APP_RESTART_POLICY` 按场景设置：离线文件批处理用 `no`，持续 RTSP 服务可用 `unless-stopped`，同时配置日志轮转与外部监控。

## 5. 生产加固

- 不暴露 PostgreSQL 到公网；默认 Compose 仅绑定 `127.0.0.1`。
- 用防火墙/安全组限制服务器，镜像和 parser 在制品库做签名与漏洞扫描。
- 不给容器 `privileged`；GPU 通过 Compose device reservation 注入。
- 将输出接入容量配额、留存策略和异步归档，避免磁盘写满。
- 监控 GPU 温度/显存/利用率、Pipeline FPS、RTSP reconnect、推理延迟、事件队列、数据库池和磁盘。
- 定期执行 pgvector 备份与恢复演练；模型版本与数据库人员库变更必须可审计。
- 先用影子流/回放做灰度，确认检测、人脸阈值、行为误报和截图合规后再切流。

## 6. 升级与回滚

升级前导出数据库备份，保留上一版镜像、配置、engine/parser 和校验值。新旧镜像不要共享正在写的结果文件。发布步骤建议：停止流量、优雅停止、备份、加载新镜像、运行固定回放、启动小流量、观察，再全量切换。

回滚时恢复**同一组**镜像 + engine + parser + config；不能只回滚 Python 镜像而继续使用新 TensorRT engine。数据库若有 schema 变更，按迁移工具的向下兼容策略处理，不直接删除生产卷。

官方资料：

- [DeepStream 9.0 installation](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_Installation.html)
- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
