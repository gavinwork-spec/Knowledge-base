# 知识库微服务架构文档

## 🎯 架构概述

本项目已从单体架构重构为事件驱动的微服务架构，提升了系统的可扩展性、响应性和可维护性。

### 🏗️ 核心架构组件

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   API Gateway   │  Event Bus      │   Load Balancer  │  Monitoring     │
│   (端口: 8000)  │  (Redis)        │                  │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────┬─────────┘
         │                │                │            │
┌─────────────────────────────────────────────────────────────────────┐
│                    Message Broker (Redis)                            │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│  Document Service │  Search Service   │  Agent Service   │  Notify Service  │
│  (端口: 8003)   │  (端口: 8004)     │  (端口: 8005)    │  (端口: 8006)   │
│                 │                 │                 │                 │
│ • PDF解析       │ • 向量检索       │ • 学习Agent     │ • 邮件/微信      │
│ • 文档提取     │ • 语义搜索       │ • 推荐Agent     │ • 系统通知      │
│ • 格式转换     │ • 关键词搜索     │ • 策略Agent     │ • 状态变更      │
│                 │                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────┴─────────┘
         │                │                │            │
┌─────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL Database (主数据库)                         │
│  - knowledge_entries, recommendations, users, events              │
│  - 向量索引 (pgvector), 全文搜索索引                              │
│  - 事件存储表, 工作流状态表                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
Knowledge base/
├── docker-compose.yml              # Docker Compose配置
├── MICROSERVICES_README.md         # 本文档
├── event_system_design.md          # 事件系统设计文档
├── microservices/                  # 微服务目录
│   ├── event_system.py            # 事件系统核心模块
│   ├── database_migration.py      # 数据库迁移脚本
│   ├── document_service.py        # 文档服务
│   ├── sql/                       # SQL脚本
│   │   └── init.sql              # 数据库初始化脚本
│   └── deploy_microservices.py    # 部署脚本
├── redis/                          # Redis配置
│   └── redis.conf                 # Redis配置文件
├── github-frontend/                # 前端应用
└── agents/                         # AI Agent配置
```

## 🚀 快速开始

### 前置要求

- Docker 和 Docker Compose
- Python 3.8+
- 至少 8GB RAM
- 20GB 可用磁盘空间

### 1. 克隆项目

```bash
git clone https://github.com/gavinwork-spec/Knowledge-base.git
cd Knowledge-base
```

### 2. 完整部署（推荐）

```bash
# 一键部署所有服务
python microservices/deploy_microservices.py full

# 或者使用Docker Compose
docker-compose up -d
```

### 3. 分步部署

```bash
# 部署基础设施（Redis + PostgreSQL）
python microservices/deploy_microservices.py deploy-infra

# 运行数据库迁移
python microservices/deploy_microservices.py migrate

# 部署微服务
python microservices/deploy_microservices.py deploy-services

# 部署监控服务（可选）
python microservices/deploy_microservices.py deploy-monitoring
```

### 4. 健康检查

```bash
# 检查所有服务状态
python microservices/deploy_microservices.py health

# 查看服务状态
python microservices/deploy_microservices.py status

# 查看特定服务日志
python microservices/deploy_microservices.py logs --service document-service
```

## 🔧 服务详情

### API Gateway (端口: 8000)
- **功能**: 统一入口，路由请求，负载均衡
- **健康检查**: http://localhost:8000/health
- **配置**: 速率限制，认证授权

### Document Service (端口: 8003)
- **功能**: 文档上传，解析，格式转换
- **支持格式**: PDF, Word, Excel, 图片, 文本
- **健康检查**: http://localhost:8003/health
- **主要API**:
  - `POST /upload` - 上传文档
  - `POST /process/{document_id}` - 处理文档
  - `GET /documents/{document_id}` - 获取文档信息

### Search Service (端口: 8004)
- **功能**: 向量搜索，语义检索，全文搜索
- **向量模型**: all-MiniLM-L6-v2 (384维)
- **健康检查**: http://localhost:8004/health
- **主要API**:
  - `GET /search` - 执行搜索
  - `POST /index` - 添加索引
  - `GET /similarity/{document_id}` - 相似度搜索

### Agent Service (端口: 8005)
- **功能**: AI Agent管理，任务执行
- **支持Agent**: 学习Agent，推荐Agent，策略Agent
- **健康检查**: http://localhost:8005/health
- **主要API**:
  - `POST /agents/{agent_type}/run` - 执行Agent
  - `GET /agents` - 列出Agent
  - `GET /tasks/{task_id}` - 查看任务状态

### Notify Service (端口: 8006)
- **功能**: 多渠道通知，邮件，微信，系统通知
- **健康检查**: http://localhost:8006/health
- **主要API**:
  - `POST /notify` - 发送通知
  - `GET /templates` - 获取通知模板

### User Service (端口: 8007)
- **功能**: 用户管理，认证授权，权限控制
- **健康检查**: http://localhost:8007/health
- **主要API**:
  - `POST /auth/login` - 用户登录
  - `POST /users` - 创建用户
  - `GET /users/{user_id}` - 获取用户信息

## 📊 监控和日志

### Prometheus (端口: 9090)
- **访问**: http://localhost:9090
- **功能**: 指标收集，查询，告警

### Grafana (端口: 3001)
- **访问**: http://localhost:3001
- **账号**: admin / admin
- **功能**: 数据可视化，仪表板

### ELK Stack
- **Elasticsearch**: http://localhost:9200
- **Kibana**: http://localhost:5601
- **功能**: 日志聚合，搜索，分析

### MinIO (端口: 9000)
- **访问**: http://localhost:9000
- **账号**: minioadmin / minioadmin123
- **功能**: 对象存储，文件管理

## 🔄 事件系统

### 事件类型

```python
class EventType(Enum):
    # 文档事件
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_PROCESSING_FAILED = "document.processing_failed"

    # 搜索事件
    SEARCH_QUERY = "search.query"
    SEARCH_COMPLETED = "search.completed"

    # Agent事件
    AGENT_TRIGGERED = "agent.triggered"
    AGENT_COMPLETED = "agent.completed"

    # 用户事件
    USER_LOGIN = "user.login"
    USER_QUERY = "user.query"

    # 系统事件
    SYSTEM_ALERT = "system.alert"
    SYSTEM_BACKUP = "system.backup"
```

### 事件流程示例

```mermaid
sequenceDiagram
    participant U as 用户
    participant G as API Gateway
    participant E as Event Bus
    participant DS as Document Service
    participant SS as Search Service

    U->>G: 上传文档
    G->>E: 发布 DOCUMENT_UPLOADED 事件
    E->>DS: 转发事件
    DS->>DS: 处理文档
    DS->>E: 发布 DOCUMENT_PROCESSED
    E->>SS: 转发事件
    SS->>SS: 更新搜索索引
```

## 🗄️ 数据库架构

### 核心表结构

- **knowledge_entries**: 知识条目主表
- **vector_index**: 向量索引表
- **events**: 事件存储表
- **users**: 用户管理表
- **chat_history**: 聊天历史表
- **search_history**: 搜索历史表
- **recommendations**: 推荐结果表
- **workflow_states**: 工作流状态表

### 向量搜索

使用 pgvector 扩展进行高效的向量相似性搜索：

```sql
-- 查找相似文档
SELECT ke.id, ke.name, ke.entity_type,
1 - (vi.vector <=> query_vector) as similarity
FROM vector_index vi
JOIN knowledge_entries ke ON vi.knowledge_entry_id = ke.id
ORDER BY vi.vector <=> query_vector
LIMIT 10;
```

## 🔧 开发指南

### 添加新服务

1. 在 `microservices/` 目录创建服务文件
2. 在 `docker-compose.yml` 添加服务定义
3. 更新 `deploy_microservices.py` 服务列表
4. 创建相应的健康检查端点

### 事件处理

```python
from microservices.event_system import EventBus, Event, EventType

# 初始化事件总线
event_bus = EventBus("redis://localhost:6379/0")
await event_bus.connect()

# 发布事件
event = Event(
    type=EventType.DOCUMENT_UPLOADED,
    source="document-service",
    data={"document_id": "123", "filename": "test.pdf"}
)
await event_bus.publish_event(event)
```

### 数据库访问

```python
import asyncpg

# 连接数据库
conn = await asyncpg.connect(
    "postgresql://postgres:postgres@localhost:5432/knowledge_base"
)

# 执行查询
result = await conn.fetch(
    "SELECT * FROM knowledge_entries WHERE entity_type = $1",
    "product"
)
```

## 📝 部署命令参考

```bash
# 完整部署
python microservices/deploy_microservices.py full

# 分步部署
python microservices/deploy_microservices.py deploy-infra
python microservices/deploy_microservices.py migrate
python microservices/deploy_microservices.py deploy-services

# 服务管理
python microservices/deploy_microservices.py status
python microservices/deploy_microservices.py health
python microservices/deploy_microservices.py logs --service <service-name>
python microservices/deploy_microservices.py restart --service <service-name>
python microservices/deploy_microservices.py stop

# Docker Compose 命令
docker-compose up -d                    # 启动所有服务
docker-compose down                     # 停止所有服务
docker-compose logs -f <service>        # 查看服务日志
docker-compose restart <service>        # 重启服务
docker-compose exec <service> bash      # 进入服务容器
```

## 🚨 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   # 检查服务日志
   python microservices/deploy_microservices.py logs --service <service-name>

   # 检查Docker状态
   docker-compose ps
   ```

2. **数据库连接失败**
   ```bash
   # 检查PostgreSQL状态
   docker-compose logs postgres

   # 重新运行迁移
   python microservices/deploy_microservices.py migrate
   ```

3. **Redis连接失败**
   ```bash
   # 检查Redis状态
   docker-compose logs redis

   # 测试连接
   redis-cli ping
   ```

4. **内存不足**
   ```bash
   # 检查资源使用
   docker stats

   # 调整Docker资源限制
   # 编辑 docker-compose.yml 中的资源限制
   ```

### 性能优化

1. **PostgreSQL优化**
   - 调整 `shared_buffers` 和 `work_mem`
   - 优化向量索引配置
   - 定期执行 `VACUUM ANALYZE`

2. **Redis优化**
   - 调整 `maxmemory` 和 `maxmemory-policy`
   - 配置适当的持久化策略
   - 监控内存使用情况

3. **应用优化**
   - 使用连接池
   - 实现缓存策略
   - 优化查询性能

## 🔐 安全配置

### 生产环境建议

1. **更改默认密码**
   - PostgreSQL 用户密码
   - Redis 认证密码
   - Grafana 管理员密码
   - MinIO 访问密钥

2. **网络安全**
   - 配置防火墙规则
   - 使用 HTTPS/TLS
   - 限制数据库访问

3. **数据加密**
   - 启用 PostgreSQL 数据加密
   - 配置 Redis TLS
   - 加密敏感配置文件

## 📈 扩展指南

### 水平扩展

```bash
# 扩展服务实例
docker-compose up -d --scale document-service=3 --scale search-service=2
```

### 集群部署

1. 使用 Docker Swarm 或 Kubernetes
2. 配置服务发现
3. 设置负载均衡器
4. 配置持久化存储

## 📞 技术支持

如遇问题，请检查：

1. 系统日志: `docker-compose logs`
2. 服务健康: `/health` 端点
3. 资源使用: `docker stats`
4. 网络连接: `docker network ls`

更多详细信息请参考：
- [事件系统设计文档](event_system_design.md)
- [API文档](http://localhost:8000/docs)
- [监控仪表板](http://localhost:3001)