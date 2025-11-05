# Knowledge Base 使用指南

## 🎯 项目概述

这是一个基于SQLite的制造业客户管理系统，能够自动扫描和导入客户资料、图纸文件，构建结构化的知识库。

## 📁 项目结构

```
Knowledge base/
├── data/
│   ├── db.sqlite                    # 主数据库文件
│   └── processed/                   # 处理日志目录
├── models.py                        # 数据库模型定义
├── setup_models.py                  # 数据库初始化脚本
├── ingest_customers.py              # 客户资料导入脚本
├── ingest_drawings.py               # 图纸资料导入脚本
├── verify_database.py               # 数据库验证脚本
├── knowledge_base_manager.py        # 综合管理工具
├── customer_ingest_agent.yaml       # 客户资料自动化代理
├── drawing_ingest_agent.yaml        # 图纸资料自动化代理
└── README.md                        # 项目说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip3 install pandas openpyxl xlrd PyPDF2 Pillow
```

### 2. 初始化数据库

```bash
python3 setup_models.py
```

### 3. 完整导入（推荐首次使用）

```bash
python3 knowledge_base_manager.py full-import
```

## 🛠️ 管理工具使用

### 查看统计信息

```bash
python3 knowledge_base_manager.py stats
```

### 搜索客户

```bash
python3 knowledge_base_manager.py search-customers --keyword "ABC"
```

### 搜索图纸

```bash
python3 knowledge_base_manager.py search-drawings --keyword "螺丝"
```

### 单独导入客户资料

```bash
python3 knowledge_base_manager.py import-customers
# 或指定目录
python3 knowledge_base_manager.py import-customers --dir "/path/to/customer/files"
```

### 单独导入图纸资料

```bash
python3 knowledge_base_manager.py import-drawings
# 或指定目录
python3 knowledge_base_manager.py import-drawings --dir "/path/to/drawing/files"
```

### 导出摘要报告

```bash
python3 knowledge_base_manager.py export
# 或指定输出文件
python3 knowledge_base_manager.py export --output "report.txt"
```

### 清理临时文件

```bash
python3 knowledge_base_manager.py cleanup
```

## 📁 监控的文件夹

### 客户资料文件夹
- **路径**: `/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户/`
- **支持格式**: Excel (.xlsx, .xls), CSV (.csv), 文本 (.txt)
- **提取信息**: 公司名称、联系人、邮箱、电话、国家等

### 图纸资料文件夹
- **路径**: `/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价/`
- **支持格式**: PDF, 图片 (JPG, PNG等), DWG, DXF等
- **提取信息**: 文件名、路径、修改时间、产品类别、尺寸等

## 🤖 自动化代理

### 客户资料监控代理 (`customer_ingest_agent.yaml`)
- 自动监控客户资料文件夹
- 检测新文件并自动导入
- 生成处理日志

### 图纸资料监控代理 (`drawing_ingest_agent.yaml`)
- 自动监控询盘询价文件夹
- 批量处理图纸文件
- 自动关联客户信息

## 📊 数据库结构

### 核心实体

1. **Customer (客户)** - 以"联系人邮件 + 公司名称"为核心标识
2. **Factory (工厂)** - 制造工厂信息
3. **Drawing (图纸记录)** - 客户图纸管理
4. **FactoryQuote (工厂报价)** - 工厂报价记录
5. **Specification (规格)** - 产品规格标准
6. **ProcessStatus (定制流程状态)** - 订单流程跟踪

### 数据关系

```
Customer (1) → (N) Drawing
Factory (1) → (N) FactoryQuote
Drawing (1) → (N) ProcessStatus
```

## 🔍 智能匹配功能

### 客户匹配
- 基于邮箱精确匹配
- 基于公司名称模糊匹配
- 防重复数据导入

### 图纸分类
- 自动从文件名提取产品类别
- 智能识别图纸编号
- 提取尺寸、材料信息

## 📈 当前数据统计

```bash
python3 knowledge_base_manager.py stats
```

最新统计（截至2025-11-05）：
- 客户记录: 2 条
- 图纸记录: 812 条
- 工厂记录: 2 条
- 报价记录: 2 条
- 规格记录: 2 条
- 流程状态: 2 条

## 🔧 故障排除

### 常见问题

1. **依赖库缺失**
   ```bash
   pip3 install pandas openpyxl xlrd PyPDF2 Pillow
   ```

2. **权限问题**
   ```bash
   chmod +x knowledge_base_manager.py
   ```

3. **数据库锁定**
   - 确保没有其他进程正在使用数据库
   - 重启相关脚本

4. **文件路径错误**
   - 检查坚果云同步状态
   - 确认文件夹路径存在

### 日志文件位置

- 客户导入日志: `./data/processed/customer_ingest_log.json`
- 图纸导入日志: `./data/processed/drawing_ingest_log.json`
- 代理活动日志: `./data/processed/agent_activity.log`
- 错误日志: `./data/processed/*_error.log`

## 🔄 定期维护

### 建议定期执行

1. **每周清理临时文件**
   ```bash
   python3 knowledge_base_manager.py cleanup
   ```

2. **每月导出统计报告**
   ```bash
   python3 knowledge_base_manager.py export
   ```

3. **数据库验证**
   ```bash
   python3 verify_database.py
   ```

## 📞 技术支持

如遇到问题，请检查：
1. 日志文件中的错误信息
2. 依赖库是否正确安装
3. 文件夹权限和路径设置
4. 网络连接状态（影响文件监控）

---

**注意**: 这是一个自动化知识库系统，建议定期检查数据质量和处理日志，确保系统正常运行。