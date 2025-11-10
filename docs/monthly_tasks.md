# 月度任务自动化脚本

## 📅 月度任务清单

### 任务1: 数据库备份
```bash
python3 backup_manager.py --type full --message "月度自动备份"
```

### 任务2: 数据质量检查
```bash
python3 data_quality_check.py > reports/monthly_quality_$(date +%Y%m).txt
```

### 任务3: 路径稳定性检查
```bash
python3 path_stability_check.py > reports/monthly_path_$(date +%Y%m).txt
```

### 任务4: 产品分类更新
```bash
python3 product_classification_manager.py
```

### 任务5: 数据库优化
```bash
python3 database_optimizer.py
```

### 任务6: 分析准备
```bash
python3 prepare_analysis.py
```

### 任务7: 清理旧文件
```bash
# 清理30天前的日志
find ./logs -name "*.log" -mtime +30 -delete

# 清理7天前的临时文件
find ./data/processed -name "*" -mtime +7 -delete

# 清理90天前的备份 (保留月度备份)
python3 backup_manager.py --cleanup 90
```

## 🔄 自动化执行脚本

创建月度任务脚本:
```bash
#!/bin/bash
# monthly_tasks.sh

echo "🚀 开始月度维护任务 - $(date)"

# 创建报告目录
mkdir -p reports/monthly_$(date +%Y%m)

# 执行所有月度任务
echo "📦 执行数据库备份..."
python3 backup_manager.py --type full --message "月度自动备份"

echo "🔍 执行数据质量检查..."
python3 data_quality_check.py > reports/monthly_$(date +%Y%m)/quality_check.txt

echo "🔍 执行路径稳定性检查..."
python3 path_stability_check.py > reports/monthly_$(date +%Y%m)/path_check.txt

echo "🏷️ 更新产品分类..."
python3 product_classification_manager.py

echo "⚡ 优化数据库..."
python3 database_optimizer.py

echo "📊 准备分析数据..."
python3 prepare_analysis.py

echo "🧹 清理旧文件..."
find ./logs -name "*.log" -mtime +30 -delete
find ./data/processed -name "*" -mtime +7 -delete
python3 backup_manager.py --cleanup 90

echo "✅ 月度维护任务完成 - $(date)"
```

## 📧 自动化通知设置

### 使用cron定时任务
```bash
# 编辑crontab
crontab -e

# 添加月度任务 (每月1日凌晨2点执行)
0 2 1 * * cd /Users/gavin/Knowledge\ base && ./monthly_tasks.sh
```

### 邮件通知配置
```bash
# 在monthly_tasks.sh末尾添加邮件通知
echo "月度维护报告已生成，请查看 reports/monthly_$(date +%Y%m)/ 目录" | mail -s "知识库月度维护完成" admin@example.com
```

## 📊 月度报告模板

每月生成的报告应包含:
1. 数据质量统计
2. 系统性能指标
3. 数据增长趋势
4. 问题及解决方案
5. 下月改进计划