# 提醒规则配置文档

## 概述

本文档定义了知识库提醒系统的17条核心提醒规则，用于监控订单状态、质量异常、交付进度等关键业务环节，确保及时响应和风险预警。

## 🚨 高优先级提醒规则

| 规则ID | 规则名称 | 触发条件 | 检查频率 | 优先级 | 通知方式 | 处理方式 | 自动化级别 |
|--------|----------|----------|----------|--------|----------|----------|------------|
| R001 | 客户图纸状态变更 | 新图纸上传、状态变更 | 每小时 | 高 | 邮件+系统通知 | 自动标记 | 完全自动化 |
| R002 | 报价超时预警 | 报价时间超过设定阈值 | 每15分钟 | 高 | 邮件+短信 | 升级到主管 | 半自动化 |
| R003 | 质量异常报警 | 质量评分低于阈值 | 实时 | 高 | 邮件+短信+系统通知 | 立即处理 | 完全自动化 |
| R004 | 交付超期预警 | 预计交付日期临近 | 每天上午9点 | 高 | 邮件+系统通知 | 协调生产 | 半自动化 |
| R005 | 客户投诉提醒 | 新投诉记录或状态更新 | 实时 | 高 | 邮件+系统通知 | 转交客服 | 完全自动化 |

## ⚡ 中优先级提醒规则

| 规则ID | 规则名称 | 触发条件 | 检查频率 | 优先级 | 通知方式 | 处理方式 | 自动化级别 |
|--------|----------|----------|----------|--------|----------|----------|------------|
| R006 | 批次生产计划提醒 | 新批次创建或计划变更 | 每小时 | 中 | 邮件+系统通知 | 更新生产排期 | 半自动化 |
| R007 | 技术参数更新提醒 | 技术参数文件更新 | 每天上午10点 | 中 | 系统通知 | 自动分发 | 完全自动化 |
| R008 | 客户跟进提醒 | 超过7天未跟进的客户 | 每天上午11点 | 中 | 系统通知 | 分配给销售 | 半自动化 |
| R009 | 报价分析报告提醒 | 报价数据更新完成 | 每天下午2点 | 中 | 邮件 | 自动生成 | 完全自动化 |
| R010 | 生产进度更新提醒 | 生产状态变更 | 每30分钟 | 中 | 系统通知 | 自动更新 | 完全自动化 |

## 📋 低优先级提醒规则

| 规则ID | 规则名称 | 触发条件 | 检查频率 | 优先级 | 通知方式 | 处理方式 | 自动化级别 |
|--------|----------|----------|----------|--------|----------|----------|------------|
| R011 | 数据备份提醒 | 数据备份完成或失败 | 每天晚上8点 | 低 | 系统通知 | 记录日志 | 完全自动化 |
| R012 | 月度统计报告提醒 | 月度统计完成 | 每月1号上午9点 | 低 | 邮件 | 自动发送 | 完全自动化 |
| R013 | 员工生日提醒 | 员工生日当天 | 每天上午8点 | 低 | 系统通知 | 自动祝福 | 完全自动化 |
| R014 | 合同到期提醒 | 合同到期前30天 | 每周一上午10点 | 低 | 邮件 | 通知法务 | 完全自动化 |
| R015 | 库存预警提醒 | 库存低于安全库存 | 每天下午3点 | 低 | 系统通知 | 建议补货 | 完全自动化 |
| R016 | 供应商评估提醒 | 供应商评估完成 | 每季度最后一天 | 低 | 邮件 | 更新评估 | 完全自动化 |
| R017 | 设备维护提醒 | 设备维护计划到期 | 每周日上午9点 | 低 | 邮件+系统通知 | 安排维护 | 半自动化 |

## 触发条件详解

### 🚨 高优先级规则触发条件

#### R001 客户图纸状态变更
**触发条件:**
```sql
-- 新图纸上传
SELECT d.id, d.drawing_name, d.created_at, c.company_name
FROM drawings d
LEFT JOIN customers c ON d.customer_id = c.id
WHERE d.created_at > datetime('now', '-1 hour')

-- 状态变更
SELECT d.id, d.drawing_name, d.status_updated_at, c.company_name
FROM drawings d
LEFT JOIN customers c ON d.customer_id = c.id
WHERE d.status_updated_at > datetime('now', '-1 hour')
```

**业务逻辑:** 新图纸上传到系统、图纸审核状态变更、图纸版本更新、图纸关联的项目状态变更时立即通知相关人员。

#### R002 报价超时预警
**触发条件:**
```sql
-- 正常订单报价超时24小时
SELECT fq.id, fq.quote_date, f.factory_name, c.company_name
FROM factory_quotes fq
JOIN factories f ON fq.factory_id = f.id
LEFT JOIN customers c ON fq.customer_id = c.id
WHERE fq.status = 'processing'
AND fq.quote_date < datetime('now', '-24 hours')
AND fq.urgent_flag = FALSE

-- 紧急订单报价超时12小时
SELECT fq.id, fq.quote_date, f.factory_name, c.company_name
FROM factory_quotes fq
JOIN factories f ON fq.factory_id = f.id
LEFT JOIN customers c ON fq.customer_id = c.id
WHERE fq.status = 'processing'
AND fq.quote_date < datetime('now', '-12 hours')
AND fq.urgent_flag = TRUE

-- 大额订单报价超时48小时
SELECT fq.id, fq.quote_date, f.factory_name, c.company_name
FROM factory_quotes fq
JOIN factories f ON fq.factory_id = f.id
LEFT JOIN customers c ON fq.customer_id = c.id
WHERE fq.status = 'processing'
AND fq.quote_date < datetime('now', '-48 hours')
AND fq.total_amount > 100000
```

**业务逻辑:** 根据订单类型设定不同的超时阈值，确保重要订单得到优先处理。

#### R003 质量异常报警
**触发条件:**
```sql
-- 质量评分低于70分
SELECT q.id, q.score, q.inspection_date, p.product_name
FROM quality_inspections q
JOIN products p ON q.product_id = p.id
WHERE q.score < 70
AND q.created_at > datetime('now', '-1 hour')

-- 连续3次质检不合格
SELECT d.id, d.drawing_name, COUNT(qi.id) as failure_count
FROM drawings d
JOIN quality_inspections qi ON d.id = qi.drawing_id
WHERE qi.result = 'FAIL'
AND qi.inspection_date > date('now', '-7 days')
GROUP BY d.id
HAVING failure_count >= 3

-- 客户投诉质量问题
SELECT comp.id, comp.complaint_date, comp.description, c.company_name
FROM complaints comp
JOIN customers c ON comp.customer_id = c.id
WHERE comp.type = 'QUALITY'
AND comp.created_at > datetime('now', '-1 hour')
```

**业务逻辑:** 实时监控质量指标，及时发现质量问题并通知相关人员进行处理。

#### R004 交付超期预警
**触发条件:**
```sql
-- 预计交付日期提前7天
SELECT po.id, po.expected_delivery_date, c.company_name
FROM production_orders po
JOIN customers c ON po.customer_id = c.id
WHERE po.expected_delivery_date BETWEEN date('now') AND date('now', '+7 days')
AND po.status IN ('in_production', 'ready')

-- 实际进度落后于计划超过20%
SELECT po.id, po.progress_percentage, po.planned_progress, c.company_name
FROM production_orders po
JOIN customers c ON po.customer_id = c.id
WHERE (po.planned_progress - po.progress_percentage) > 20
AND po.status = 'in_production'

-- 关键工序延误
SELECT ps.id, ps.process_name, ps.planned_completion, ps.actual_completion, po.id as order_id
FROM production_schedule ps
JOIN production_orders po ON ps.order_id = po.id
WHERE ps.planned_completion < datetime('now')
AND ps.actual_completion IS NULL
AND ps.is_critical = TRUE
```

**业务逻辑:** 提前预警交付风险，确保有充足时间采取纠正措施。

#### R005 客户投诉提醒
**触发条件:**
```sql
-- 新投诉记录创建
SELECT comp.id, comp.complaint_date, comp.type, c.company_name
FROM complaints comp
JOIN customers c ON comp.customer_id = c.id
WHERE comp.created_at > datetime('now', '-1 hour')

-- 投诉处理状态更新
SELECT comp.id, comp.status, comp.updated_at, c.company_name
FROM complaints comp
JOIN customers c ON comp.customer_id = c.id
WHERE comp.updated_at > datetime('now', '-1 hour')

-- 投诉升级处理
SELECT comp.id, comp.escalation_level, comp.escalation_date, c.company_name
FROM complaints comp
JOIN customers c ON comp.customer_id = c.id
WHERE comp.escalation_level > 1
AND comp.escalation_date > datetime('now', '-1 hour')

-- 投诉解决超时
SELECT comp.id, comp.complaint_date, comp.resolution_deadline, c.company_name
FROM complaints comp
JOIN customers c ON comp.customer_id = c.id
WHERE comp.status IN ('pending', 'investigating')
AND comp.resolution_deadline < datetime('now')
```

**业务逻辑:** 确保客户投诉得到及时响应和处理，避免客户满意度下降。

### ⚡ 中优先级规则触发条件

#### R006 批次生产计划提醒
**触发条件:**
```sql
-- 新批次创建
SELECT pb.id, pb.batch_number, pb.created_at, p.product_name
FROM production_batches pb
JOIN products p ON pb.product_id = p.id
WHERE pb.created_at > datetime('now', '-1 hour')

-- 计划变更
SELECT pb.id, pb.batch_number, pb.updated_at, p.product_name
FROM production_batches pb
JOIN products p ON pb.product_id = p.id
WHERE pb.updated_at > datetime('now', '-1 hour')
AND pb.status != pb.last_status
```

#### R007 技术参数更新提醒
**触发条件:**
```sql
-- 技术参数文件更新
SELECT tp.id, tp.parameter_name, tp.updated_at, p.product_name
FROM technical_parameters tp
JOIN products p ON tp.product_id = p.id
WHERE tp.updated_at > datetime('now', '-24 hours')
```

#### R008 客户跟进提醒
**触发条件:**
```sql
-- 超过7天未跟进的客户
SELECT c.id, c.company_name, c.last_contact_date, c.contact_email
FROM customers c
WHERE c.last_contact_date < date('now', '-7 days')
AND c.status = 'active'
```

#### R009 报价分析报告提醒
**触发条件:**
```sql
-- 报价数据更新完成
SELECT COUNT(*) as updated_quotes
FROM factory_quotes
WHERE updated_at > date('now', '-1 day')
```

#### R010 生产进度更新提醒
**触发条件:**
```sql
-- 生产状态变更
SELECT po.id, po.status, po.updated_at, c.company_name
FROM production_orders po
JOIN customers c ON po.customer_id = c.id
WHERE po.updated_at > datetime('now', '-30 minutes')
```

### 📋 低优先级规则触发条件

#### R011-R017 详细触发条件类似，按业务需求设定相应的时间阈值和条件

## 通知方式配置

### 邮件通知配置
- **SMTP服务器**: smtp.company.com
- **端口**: 587
- **发件人**: system@company.com
- **模板**: HTML格式，包含详细信息和操作链接

### 短信通知配置
- **服务商**: 阿里云短信服务
- **模板**: 简洁明了，包含关键信息
- **频率限制**: 同一规则每小时最多1条

### 系统通知配置
- **前端实时推送**: WebSocket连接
- **通知中心**: 系统内消息中心
- **移动端推送**: 企业微信/钉钉集成

## 自动化处理流程

### 完全自动化流程
1. **规则触发** → 2. **条件检查** → 3. **自动处理** → 4. **结果通知** → 5. **记录日志**

### 半自动化流程
1. **规则触发** → 2. **条件检查** → 3. **人工审核** → 4. **执行处理** → 5. **结果通知** → 6. **记录日志**

## 系统配置参数

### 全局配置
```yaml
reminder_system:
  enabled: true
  max_daily_notifications: 100
  notification_rate_limit: 10  # 每小时最大通知数
  default_timezone: "Asia/Shanghai"
  log_retention_days: 90

email_config:
  smtp_server: "smtp.company.com"
  smtp_port: 587
  use_tls: true
  sender_email: "system@company.com"
  sender_name: "知识库提醒系统"

sms_config:
  provider: "aliyun"
  access_key: "your_access_key"
  secret_key: "your_secret_key"
  template_id: "SMS_123456789"

webhook_config:
  slack_webhook_url: "https://hooks.slack.com/..."
  dingtalk_webhook_url: "https://oapi.dingtalk.com/..."
```

### 规则特定配置
```yaml
rule_configs:
  R002:
    timeout_thresholds:
      normal: 24  # 小时
      urgent: 12  # 小时
      large_order: 48  # 小时
    escalation_enabled: true
    escalation_users: ["manager@company.com"]

  R003:
    quality_threshold: 70
    consecutive_failures: 3
    auto_investigation: true

  R004:
    advance_warning_days: 7
    progress_delay_threshold: 20  # 百分比
    critical_processes: ["cutting", "assembly", "quality_check"]
```

## 数据库字段要求

### 提醒记录表结构
```sql
CREATE TABLE reminder_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id VARCHAR(10) NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    trigger_time DATETIME NOT NULL,
    trigger_condition TEXT NOT NULL,
    priority VARCHAR(10) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    assigned_to VARCHAR(50),
    due_time DATETIME,
    completed_time DATETIME,
    notification_methods VARCHAR(100),
    auto_processed BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 索引配置
```sql
CREATE INDEX idx_reminder_rule_id ON reminder_records(rule_id);
CREATE INDEX idx_reminder_status ON reminder_records(status);
CREATE INDEX idx_reminder_priority ON reminder_records(priority);
CREATE INDEX idx_reminder_trigger_time ON reminder_records(trigger_time);
CREATE INDEX idx_reminder_assigned_to ON reminder_records(assigned_to);
```

## 监控和报告

### 系统监控指标
- 每日提醒数量统计
- 规则触发频率分析
- 平均处理时间统计
- 自动化处理成功率
- 用户响应时间分析

### 报告生成
- 每日提醒执行报告
- 每周规则效果分析
- 每月系统性能报告
- 异常情况专项报告

## 更新维护

### 规则更新流程
1. **需求评估** → 2. **规则设计** → 3. **测试验证** → 4. **发布上线** → 5. **效果监控**

### 版本控制
- 规则配置版本化
- 变更记录追踪
- 回滚机制支持
- 测试环境验证

## 使用说明

### 管理员操作
1. **规则配置**: 通过后台管理系统修改规则参数
2. **用户管理**: 配置通知接收人和权限
3. **监控查看**: 实时监控系统运行状态
4. **报告导出**: 定期导出分析报告

### 普通用户操作
1. **提醒查看**: 在系统中查看分配给自己的提醒
2. **状态更新**: 更新提醒处理状态
3. **备注添加**: 为提醒添加处理说明
4. **通知设置**: 个人通知偏好设置

---

**文档版本**: v2.0
**最后更新**: 2025-11-06
**维护人员**: 系统管理员
**审核状态**: 已审核
