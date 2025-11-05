# 提醒规则配置

## 📋 概述

本文档定义了业务提醒规则，用于自动监控关键业务指标并及时通知相关人员。规则基于 AYA Fastener、Homelux 和 Sinostar 的业务特点制定。

## 🔔 提醒规则分类

### 1. 客户关系提醒

#### 1.1 长期未活跃客户
```yaml
rule: inactive_customer
trigger:
  condition: "客户超过30天无新活动"
  query: |
    SELECT c.id, c.company_name, c.contact_email,
           MAX(d.upload_date) as last_activity
    FROM customers c
    LEFT JOIN drawings d ON c.id = d.customer_id
    GROUP BY c.id
    HAVING last_activity < date('now', '-30 days') OR last_activity IS NULL

severity: medium
actions:
  - email: "向客户发送问候邮件"
  - task: "安排客户回访"
  - note: "记录客户状态更新"

message_template: |
  客户 {company_name} 已超过30天未活跃，
  最后活动时间: {last_activity}
  请安排跟进联系。
```

#### 1.3 客户询盘无下单
```yaml
rule: inquiry_no_order
trigger:
  condition: "客户连续3次询盘无下单"
  query: |
    SELECT c.company_name, COUNT(DISTINCT d.id) as inquiry_count,
           COUNT(DISTINCT ps.id) as order_count
    FROM customers c
    LEFT JOIN drawings d ON c.id = d.customer_id
    LEFT JOIN process_status ps ON c.id = ps.customer_id AND ps.status = 'batch_production'
    GROUP BY c.id
    HAVING inquiry_count >= 3 AND order_count = 0

severity: high
actions:
  - email: "发送特价优惠"
  - call: "电话跟进"
  - discount: "提供5%折扣"

message_template: |
  客户 {company_name} 已有{inquiry_count}次询盘但无下单，
  建议提供特别优惠促进转化。
```

### 2. 价格监控提醒

#### 2.1 报价上涨超过10%
```yaml
rule: price_increase_alert
trigger:
  condition: "同一产品报价较历史平均价上涨超过10%"
  query: |
    WITH recent_prices AS (
      SELECT product_category, AVG(price) as recent_avg
      FROM factory_quotes
      WHERE quote_date >= date('now', '-30 days')
      GROUP BY product_category
    ),
    historical_prices AS (
      SELECT product_category, AVG(price) as historical_avg
      FROM factory_quotes
      WHERE quote_date < date('now', '-90 days')
      GROUP BY product_category
    )
    SELECT r.product_category, r.recent_avg, h.historical_avg,
           (r.recent_avg - h.historical_avg) / h.historical_avg * 100 as price_change
    FROM recent_prices r
    JOIN historical_prices h ON r.product_category = h.product_category
    WHERE (r.recent_avg - h.historical_avg) / h.historical_avg * 100 > 10

severity: high
actions:
  - review: "重新评估价格策略"
  - notify: "通知销售团队"
  - analysis: "分析价格波动原因"

message_template: |
  警告：{product_category} 价格上涨 {price_change:.1f}%
  当前价格: {recent_avg}
  历史价格: {historical_avg}
  请及时关注！
```

#### 2.2 工厂报价差异过大
```yaml
rule: factory_price_variance
trigger:
  condition: "同一产品不同工厂报价差异超过20%"
  query: |
    SELECT product_category,
           MAX(price) as max_price,
           MIN(price) as min_price,
           (MAX(price) - MIN(price)) / MIN(price) * 100 as variance
    FROM factory_quotes
    WHERE quote_date >= date('now', '-60 days')
    GROUP BY product_category
    HAVING variance > 20

severity: medium
actions:
  - negotiate: "与工厂重新谈判价格"
  - benchmark: "重新进行价格基准测试"
  - decision: "选择最优供应商"

message_template: |
  {product_category} 不同工厂报价差异 {variance:.1f}%
  最高价: {max_price}
  最低价: {min_price}
  建议重新评估供应商选择。
```

### 3. 库存和交付提醒

#### 3.1 MOQ变更提醒
```yaml
rule: moq_change_alert
trigger:
  condition: "工厂最小起订量发生变化"
  query: |
    SELECT fq1.product_category, f.factory_name,
           fq1.moq as new_moq, fq1.quote_date as new_date,
           fq2.moq as old_moq, fq2.quote_date as old_date
    FROM factory_quotes fq1
    JOIN factory_quotes fq2 ON fq1.product_category = fq2.product_category
                        AND fq1.factory_id = fq2.factory_id
                        AND fq1.quote_date > fq2.quote_date
    WHERE fq1.quote_date >= date('now', '-7 days')
      AND fq1.moq != fq2.moq

severity: low
actions:
  - update: "更新产品目录"
  - inform: "通知销售团队"
  - review: "重新评估库存策略"

message_template: |
  {factory_name} 的 {product_category} 最小起订量变更：
  {old_moq} → {new_moq}
  变更日期: {new_date}
```

### 4. 数据质量提醒

#### 4.1 未分类图纸过多
```yaml
rule: unclassified_drawings
trigger:
  condition: "未分类图纸超过100个"
  query: |
    SELECT COUNT(*) as count
    FROM drawings
    WHERE product_category = '未分类'

severity: medium
threshold: 100
actions:
  - classify: "安排人工分类"
  - improve: "改进自动分类算法"
  - review: "定期审查分类规则"

message_template: |
  当前有 {count} 个图纸未分类，
  超过阈值 {threshold}，
  请及时处理以提高数据质量。
```

#### 4.2 客户关联率过低
```yaml
rule: low_customer_linkage
trigger:
  condition: "图纸客户关联率低于50%"
  query: |
    SELECT
      COUNT(*) as total_drawings,
      COUNT(customer_id) as linked_drawings,
      ROUND(COUNT(customer_id) * 100.0 / COUNT(*), 1) as linkage_rate
    FROM drawings

severity: low
threshold: 50
actions:
  - enhance: "增强自动匹配算法"
  - manual: "手动关联重要客户"
  - monitor: "定期监控关联进度"

message_template: |
  当前图纸客户关联率: {linkage_rate}% ({linked_drawings}/{total_drawings})
  低于目标值 {threshold}%，
  建议加强客户关联工作。
```

### 5. 业务流程提醒

#### 5.1 流程状态超时
```yaml
rule: process_timeout
trigger:
  condition: "流程状态超过预期时间"
  query: |
    SELECT ps.id, c.company_name, d.drawing_name,
           ps.status, ps.last_update_date
    FROM process_status ps
    JOIN customers c ON ps.customer_id = c.id
    JOIN drawings d ON ps.drawing_id = d.id
    WHERE ps.last_update_date < date('now', '-14 days')
      AND ps.status NOT IN ('completed', 'cancelled')

severity: medium
actions:
  - follow_up: "跟进流程进度"
  - escalate: "升级给主管"
  - update: "更新状态信息"

message_template: |
  客户 {company_name} 的流程状态异常：
  图纸: {drawing_name}
  状态: {status} (已{days}天未更新)
  请及时跟进。
```

## ⚙️ 提醒系统配置

### 执行频率
- **每小时检查**: 价格波动、流程超时
- **每日检查**: 客户活跃度、数据质量
- **每周检查**: MOQ变更、客户关联率
- **每月检查**: 整体业务趋势

### 通知渠道
- **邮件**: 重要业务提醒
- **企业微信**: 日常业务通知
- **系统内通知**: 数据质量提醒
- **短信**: 紧急业务问题

### 处理流程
1. **检测**: 系统自动检测触发条件
2. **评估**: 计算严重程度和影响范围
3. **通知**: 通过指定渠道发送提醒
4. **记录**: 记录提醒历史和处理状态
5. **跟进**: 跟踪提醒处理结果

## 📊 提醒效果监控

### 关键指标
- **提醒响应时间**: 从发送到处理的平均时间
- **问题解决率**: 成功解决的问题比例
- **业务影响**: 提醒带来的业务价值
- **误报率**: 不必要的提醒比例

### 报告模板
```yaml
weekly_report:
  period: "过去7天"
  metrics:
    - total_alerts: "总提醒数"
    - resolved_issues: "已解决问题"
    - avg_response_time: "平均响应时间"
    - business_impact: "业务影响评分"

action_items:
  - "优化高频提醒规则"
  - "改进通知渠道配置"
  - "培训团队处理流程"
```

## 🔧 自定义提醒规则

### 添加新规则
1. 在 `knowledge/reminder_rules.md` 中定义规则
2. 更新 `prepare_analysis.py` 中的检测逻辑
3. 配置通知渠道和处理流程
4. 测试规则触发条件

### 规则最佳实践
- **明确触发条件**: 避免模糊的判断标准
- **合理的严重程度**: 区分业务重要性
- **可执行的行动**: 提供具体的处理建议
- **避免过度提醒**: 防止提醒疲劳

---

**文档维护**: 根据业务发展定期更新提醒规则
**最后更新**: 2025-11-05
**维护团队**: Knowledge Base Team