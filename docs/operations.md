# 运维手册

## 常用命令

```bash
docker compose up -d --no-build
docker compose ps
docker compose logs -f --tail=200 app
docker compose stop app
docker compose down
```

离线文件任务应等待 EOS 正常退出，让 `qtmux` 完成 MP4 索引；不要直接 `docker kill`。持续 RTSP 任务停止时先执行 `docker compose stop -t 30 app`。

## 健康与 GPU

```bash
docker inspect --format '{{json .State.Health}}' "$(docker compose ps -q app)"
docker compose exec app nvidia-smi
docker compose exec app gst-inspect-1.0 nvstreammux
docker compose exec app deepstream-app --version-all
```

健康检查只说明进程/运行时就绪，不代表模型业务精度或每路 RTSP 都正常。外部监控还应验证每个 `camera_id` 的最后帧时间、输入/输出 FPS、事件队列深度和错误计数。

## 数据库备份

```bash
mkdir -p database/backups
docker compose exec -T postgres pg_dump \
  -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc \
  > "database/backups/deepstream-$(date +%Y%m%d-%H%M%S).dump"
```

变量应由当前安全会话提供，不要把密码写入脚本。定期在隔离实例执行 `pg_restore` 演练并核对向量条数、worker ID 和抽样匹配结果。

## 输出留存

`output/result.mp4` 在下次同名任务前应归档或改名；截图按事件策略清理。生产建议输出到独立持久卷/对象存储落盘器，设置磁盘阈值告警。删除前先确认法务留存与事件审计要求。

## 配置变更

每次变更保存 config/model/parser/image 的版本和 checksum，用固定回放集执行测试。RTSP 密码轮换只更新 secret 环境，不提交 Git。修改模型开关需要重启进程，因为 GStreamer 图在启动时构建。

