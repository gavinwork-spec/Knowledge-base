# 知识库系统概览文档

## 📋 目录
1. [系统简介](#系统简介)
2. [数据库架构](#数据库架构)
3. [实体说明](#实体说明)
4. [字段用途](#字段用途)
5. [基本查询示例](#基本查询示例)
6. [业务规则](#业务规则)
7. [数据导入流程](#数据导入流程)
8. [维护指南](#维护指南)

## 🚀 系统简介

### 系统目的
制造业客户管理知识库系统，专注于"联系邮箱 + 公司名称"作为核心客户标识，支持多维度数据分析。

### 核心功能
- **客户管理**: 以邮箱+公司名为核心的客户标识体系
- **图纸管理**: 产品图纸的存储、分类和关联
- **工厂管理**: 供应商工厂信息管理
- **报价管理**: 工厂报价数据追踪
- **规格管理**: 产品规格标准化
- **流程跟踪**: 业务流程状态管理

### 技术架构
- **数据库**: SQLite 3.x
- **编程语言**: Python 3.x
- **主要依赖**: pandas, sqlite3, pathlib, openpyxl

## 🗄️ 数据库架构

### 核心实体关系图
```
┌─────────────────┐
│    customers    │
├─────────────────┤
│ id (PK)         │
│ company_name    │ ◄──┐
│ contact_email   │    │
│ country         │    │
│ language        │    │
└─────────────────┘    │
         │              │
         │              │
         ▼              │
┌─────────────────┐    │
│    drawings     │    │
├─────────────────┤    │
│ id (PK)         │    │
│ customer_id (FK)│◄───┘
│ drawing_name    │
│ product_category│
│ file_path       │
└─────────────────┘

┌─────────────────┐
│   factories     │
├─────────────────┤
│ id (PK)         │ ◄──┐
│ factory_name    │    │
│ location        │    │
│ capability      │    │
└─────────────────┘    │
         │              │
         │              │
         ▼              │
┌─────────────────┐    │
│ factory_quotes  │    │
├─────────────────┤    │
│ id (PK)         │    │
│ factory_id (FK) │◄───┘
│ product_category│
│ price           │
│ quote_date      │
└─────────────────┘
```

## 📊 实体说明

### 1. customers (客户表)
**业务含义**: 存储所有客户信息，以联系邮箱和公司名作为核心标识。

**关键字段**:
- `company_name`: 公司名称 (必填)
- `contact_email`: 联系邮箱 (必填，核心标识)
- `country`: 国家/地区
- `language`: 首选语言

**业务规则**:
- company_name + contact_email 组合必须唯一
- 邮箱格式需要验证
- 支持多语言客户信息

**索引策略**:
- 单列索引: company_name, contact_email
- 复合索引: (company_name, contact_email)
- 地理索引: country

### 2. drawings (图纸表)
**业务含义**: 存储产品图纸信息，支持客户关联和产品分类。

**关键字段**:
- `customer_id`: 关联客户ID
- `drawing_name`: 图纸名称
- `product_category`: 产品分类 (自动分类结果)
- `file_path`: 文件存储路径
- `upload_date`: 上传日期

**业务规则**:
- 每个图纸必须关联到客户 (可为空)
- 支持symlink路径
- 自动产品分类

**索引策略**:
- 客户索引: customer_id
- 分类索引: product_category
- 时间索引: upload_date
- 复合索引: (customer_id, product_category)

### 3. factories (工厂表)
**业务含义**: 供应商/工厂信息管理。

**关键字段**:
- `factory_name`: 工厂名称
- `location`: 地理位置
- `capability`: 生产能力描述

### 4. factory_quotes (工厂报价表)
**业务含义**: 工厂报价信息追踪。

**关键字段**:
- `factory_id`: 关联工厂ID
- `product_category`: 产品类别
- `price`: 报价
- `quote_date`: 报价日期
- `moq`: 最小起订量

### 5. specifications (产品规格表)
**业务含义**: 产品规格标准化信息。

**关键字段**:
- `product_category`: 产品类别
- `material`: 材料规格
- `standard_or_custom`: 标准/定制标识
- `surface_treatment`: 表面处理

### 6. process_status (流程状态表)
**业务含义**: 业务流程状态跟踪。

**关键字段**:
- `drawing_id`: 关联图纸ID
- `customer_id`: 关联客户ID
- `current_stage`: 当前阶段
- `status`: 状态
- `last_update_date`: 最后更新

## 🔧 字段用途详解

### 核心标识字段
- `customers.contact_email`: 客户主要联系方式，用于客户去重和查找
- `customers.company_name`: 客户公司名称，配合邮箱形成唯一标识
- `drawings.customer_id`: 图纸客户关联，支持一对多关系

### 分类字段
- `drawings.product_category`: 自动分类结果，支持多级分类体系
  - 紧固件: 螺栓螺钉、螺母、垫圈等
  - 家具: 办公、民用、户外等
  - 建材: 基础、装饰、专用等

### 时间字段
- `customers.first_contact_date`: 首次联系时间
- `drawings.upload_date`: 图纸上传时间
- `factory_quotes.quote_date`: 报价时间
- `process_status.last_update_date`: 状态更新时间

## 📝 基本查询示例

### 客户查询
```sql
-- 按邮箱查找客户
SELECT * FROM customers WHERE contact_email = 'info@example.com';

-- 按公司名查找客户
SELECT * FROM customers WHERE company_name LIKE '%快螺%';

-- 核心标识查询 (邮箱+公司名)
SELECT * FROM customers
WHERE company_name = 'ABC Company' AND contact_email = 'info@abc.com';

-- 按地区统计客户
SELECT country, COUNT(*) as customer_count
FROM customers
GROUP BY country;
```

### 图纸查询
```sql
-- 按客户查询图纸
SELECT d.* FROM drawings d
JOIN customers c ON d.customer_id = c.id
WHERE c.company_name = 'ABC Company';

-- 按产品分类查询
SELECT product_category, COUNT(*) as count
FROM drawings
WHERE product_category IS NOT NULL
GROUP BY product_category;

-- 客户产品分类统计
SELECT
    c.company_name,
    d.product_category,
    COUNT(d.id) as drawing_count
FROM customers c
LEFT JOIN drawings d ON c.id = d.customer_id
GROUP BY c.id, d.product_category;
```

### 复合查询
```sql
-- 客户活跃度分析
SELECT
    c.company_name,
    c.contact_email,
    COUNT(d.id) as total_drawings,
    MAX(d.upload_date) as last_activity,
    COUNT(DISTINCT d.product_category) as category_count
FROM customers c
LEFT JOIN drawings d ON c.id = d.customer_id
GROUP BY c.id
ORDER BY total_drawings DESC;

-- 产品分类趋势分析
SELECT
    product_category,
    DATE(upload_date) as upload_day,
    COUNT(*) as daily_count
FROM drawings
WHERE upload_date >= date('now', '-30 days')
GROUP BY product_category, DATE(upload_date)
ORDER BY upload_day DESC;
```

## 📋 业务规则

### 客户标识规则
1. **核心标识**: company_name + contact_email
2. **唯一性**: 同一公司同一邮箱只能有一个客户记录
3. **标准化**: 邮箱格式需要验证，公司名去空格处理

### 图纸关联规则
1. **必需关联**: 图纸应关联到具体客户
2. **分类规则**: 基于文件名自动分类，支持手动调整
3. **文件路径**: 支持symlink，需要定期验证有效性

### 数据质量规则
1. **完整性**: 必填字段不能为空
2. **一致性**: 关联数据必须存在
3. **时效性**: 定期更新状态信息

## 🔄 数据导入流程

### 客户数据导入
1. **文件扫描**: 自动扫描指定目录的Excel/CSV文件
2. **数据提取**: 智能识别客户信息字段
3. **去重处理**: 基于邮箱+公司名进行去重
4. **数据验证**: 验证邮箱格式和必填字段
5. **批量插入**: 安全插入数据库

### 图纸数据导入
1. **文件扫描**: 扫描PDF、图片、DWG等文件
2. **元数据提取**: 提取文件名、路径、时间等信息
3. **客户匹配**: 基于文件名匹配客户
4. **自动分类**: 使用关键词进行产品分类
5. **路径验证**: 验证文件路径可访问性

### 错误处理
- **失败文件**: 移动到failed目录
- **错误日志**: 详细记录错误信息
- **重试机制**: 支持自动重试
- **备份策略**: 导入前自动备份

## 🔧 维护指南

### 日常维护任务
1. **数据质量检查**: 每周运行质量检查脚本
2. **路径验证**: 每月检查symlink路径有效性
3. **备份管理**: 定期创建数据库备份
4. **日志清理**: 定期清理旧日志文件

### 性能优化
1. **索引维护**: 定期分析查询性能
2. **数据库优化**: 运行VACUUM和ANALYZE命令
3. **统计更新**: 更新表统计信息

### 监控指标
- **数据完整性**: 必填字段完整率
- **关联完整性**: 外键关联完整性
- **路径有效性**: 文件可访问性
- **分类准确性**: 自动分类准确率

### 故障排除
1. **数据库锁定**: 检查长时间运行的查询
2. **路径问题**: 验证symlink有效性
3. **内存使用**: 监控数据库大小
4. **权限问题**: 检查文件访问权限

---

## 📞 技术支持

如有问题，请参考:
1. 日志文件: `./logs/`
2. 错误记录: `./data/failed/`
3. 备份文件: `./data/backups/`
4. 性能报告: `./reports/`

**文档版本**: 1.0
**最后更新**: 2025-11-05
**维护者**: Knowledge Base System