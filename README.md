# Knowledge Base - 制造业客户管理系统

这是一个基于SQLite的制造业客户管理系统数据库，包含了客户、工厂、图纸、报价、规格和流程状态等核心实体。

## 系统架构

### 核心实体

1. **Customer (客户)** - 以"联系人邮件 + 公司名称"为核心标识
2. **Factory (工厂)** - 制造工厂信息
3. **Drawing (图纸记录)** - 客户图纸管理
4. **FactoryQuote (工厂报价)** - 工厂报价记录
5. **Specification (规格)** - 产品规格标准
6. **ProcessStatus (定制流程状态)** - 订单流程跟踪

### 数据库结构

```
Knowledge base/
├── data/
│   └── db.sqlite          # SQLite数据库文件
├── models.py              # 数据库模型定义
├── setup_models.py        # 数据库初始化脚本
└── README.md             # 项目说明文档
```

## 快速开始

### 1. 初始化数据库

```bash
cd "Knowledge base"
python setup_models.py
```

这将：
- 创建SQLite数据库文件 `./data/db.sqlite`
- 初始化所有表结构
- 插入示例数据用于测试
- 验证数据库设置

### 2. 使用数据库模型

```python
from models import DatabaseManager, Customer, Factory

# 创建数据库管理器
db_manager = DatabaseManager("./data/db.sqlite")

# 使用客户实体
customer = Customer(db_manager)

# 创建新客户（核心：公司名称 + 邮箱）
customer_id = customer.create(
    company_name="ABC制造有限公司",
    contact_email="john@abc.com",
    contact_name="John Smith",
    country="美国",
    language="英语"
)

# 查询客户（核心查询方法）
customer_info = customer.get_by_company_and_email(
    "ABC制造有限公司",
    "john@abc.com"
)
```

## 实体详情

### Customer (客户)
- `id` - 主键
- `company_name` - 公司名称 (必填)
- `contact_email` - 联系人邮箱 (核心字段)
- `contact_name` - 联系人名称
- `country` - 国家/地区
- `language` - 语言偏好
- `phone` - 电话
- `first_contact_date` - 首次联系日期
- `notes` - 备注

### Factory (工厂)
- `id` - 主键
- `factory_name` - 工厂名称
- `location` - 地理位置
- `capability` - 生产能力
- `cost_reference` - 成本参考
- `production_cycle` - 生产周期
- `notes` - 备注

### Drawing (图纸记录)
- `id` - 主键
- `drawing_name` - 图纸名称
- `customer_id` - 客户ID (外键)
- `product_category` - 产品类别
- `file_path` - 文件路径
- `upload_date` - 上传日期
- `status` - 状态
- `notes` - 备注

### FactoryQuote (工厂报价)
- `id` - 主键
- `factory_id` - 工厂ID (外键)
- `product_category` - 产品类别
- `quote_date` - 报价日期
- `price` - 价格
- `moq` - 最小起订量
- `notes` - 备注

### Specification (规格)
- `id` - 主键
- `product_category` - 产品类别
- `material` - 材料
- `standard_or_custom` - 标准/定制
- `surface_treatment` - 表面处理
- `default_moq` - 默认最小起订量
- `notes` - 备注

### ProcessStatus (定制流程状态)
- `id` - 主键
- `drawing_id` - 图纸ID (外键)
- `customer_id` - 客户ID (外键)
- `status` - 状态 (图纸确认、样品、批量)
- `last_update_date` - 最后更新日期
- `notes` - 备注

## 状态选项

### 流程状态 (ProcessStatus.status)
- `drawing_confirmation` - 图纸确认
- `sample` - 样品
- `batch_production` - 批量

### 规格类型 (Specification.standard_or_custom)
- `standard` - 标准
- `custom` - 定制

## 数据库特性

- **索引优化**: 为常用查询字段创建了索引
- **外键关系**: 建立了逻辑上的外键关系
- **时间戳**: 自动记录创建和更新时间
- **核心查询**: 支持按"公司名称+邮箱"进行客户查询

## 扩展建议

1. **添加API接口**: 创建RESTful API用于数据访问
2. **用户界面**: 开发Web或桌面应用界面
3. **数据导入/导出**: 支持Excel、CSV格式
4. **报表功能**: 生成各类统计报表
5. **通知系统**: 流程状态变更通知
6. **文件管理**: 图纸文件的上传和管理

## 技术要求

- Python 3.7+
- SQLite3 (Python内置)
- 无需额外依赖

## 许可证

此项目仅用于学习和内部使用。