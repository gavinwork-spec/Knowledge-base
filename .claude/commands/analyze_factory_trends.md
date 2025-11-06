# 工厂报价趋势分析命令

## 命令概述
触发工厂报价趋势分析脚本，从工厂报价表中提取和分析价格趋势数据。

## 使用方法
```
/analyze_factory_trends
```

## 参数选项
- `--period`: 分析周期（month/quarter/year，默认month）
- `--factory`: 指定工厂ID或名称分析特定工厂
- `--category`: 指定产品类别分析特定产品
- `--export-only`: 仅导出现有数据，不重新分析

## 功能描述
1. **价格趋势分析**: 计算月度/季度价格变动趋势
2. **工厂表现评估**: 分析各工厂的报价表现
3. **价格波动性检测**: 识别价格异常和波动
4. **多维分析**: 按工厂、产品、时间维度分析
5. **结果导出**: 生成CSV和JSON格式报告

## 分析指标
- **平均价格**: 各期平均报价
- **价格变动率**: 环比价格变化百分比
- **价格区间**: 最高价、最低价、标准差
- **报价频率**: 各期报价次数
- **工厂活跃度**: 工厂综合表现指标

## 输出文件
- `factory_quote_monthly_trends_*.csv`: 月度趋势数据
- `factory_quote_quarterly_trends_*.csv`: 季度趋势数据
- `factory_quote_analysis_report_*.json`: 综合分析报告
- `price_anomalies_*.csv`: 价格异常记录

## 示例输出
```
✅ 工厂报价趋势分析完成!
📊 处理报价记录: 2
🏭 分析工厂数: 2
📦 分析产品类别: 2
⚠️ 发现异常: 0 个
⏱️ 处理时间: 0.01秒

📄 导出文件:
  monthly_trends: data/processed/factory_quote_monthly_trends_20251105_172815.csv
  quarterly_trends: data/processed/factory_quote_quarterly_trends_20251105_172815.csv
  analysis_report: data/processed/factory_quote_analysis_report_20251105_172815.json
```

## 数据要求
- 需要有效的工厂报价记录
- 报价价格必须大于0
- 报价日期不能为空

## 异常检测
- 使用IQR方法检测价格异常值
- 价格变动超过±30%标记为异常
- 缺失关键数据的记录会被跳过

## 相关文档
- `analyze_factory_quote_trends.py`: 分析脚本
- `knowledge/analysis_indicators.md`: 指标定义文档
- `data/processed/`: 分析结果输出目录

## 定时任务配置
建议配置为每月第1天自动执行，分析上月报价趋势。