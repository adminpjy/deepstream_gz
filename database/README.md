# 数据库

`database/init/` 仅在 PostgreSQL 数据卷首次创建时执行：

- `001_extensions.sql` 启用 `pgvector`。
- `002_schema.sql` 创建 512 维人脸向量表及事件表。

已有数据卷不会自动重跑迁移。结构变更应使用正式迁移工具；开发环境需要重建空库时，可在确认无数据后执行 `docker compose down -v`。生产密码只通过环境变量或 secret manager 注入，不写入 SQL、YAML 或镜像。

