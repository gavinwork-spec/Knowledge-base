#!/usr/bin/env python3
"""
验证产品分类结果的准确性
"""

import sqlite3
import pandas as pd
from collections import Counter

def verify_classification():
    """验证分类结果"""
    db_path = "./data/db.sqlite"

    # 连接数据库
    conn = sqlite3.connect(db_path)

    # 获取所有图纸的分类结果
    df = pd.read_sql_query("""
        SELECT drawing_name, product_category, customer_id, upload_date
        FROM drawings
        ORDER BY upload_date DESC
    """, conn)

    print("🔍 产品分类验证报告")
    print("=" * 60)

    # 1. 总体统计
    total_drawings = len(df)
    classified_drawings = len(df[df['product_category'] != '未分类'])
    unclassified_drawings = total_drawings - classified_drawings

    print(f"📊 总体统计:")
    print(f"  总图纸数: {total_drawings}")
    print(f"  已分类: {classified_drawings} ({classified_drawings/total_drawings*100:.1f}%)")
    print(f"  未分类: {unclassified_drawings} ({unclassified_drawings/total_drawings*100:.1f}%)")
    print()

    # 2. 分类分布
    category_stats = df['product_category'].value_counts()
    print("📈 分类分布:")
    for category, count in category_stats.items():
        percentage = count / total_drawings * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    print()

    # 3. 详细分类验证
    print("🔬 详细分类验证:")

    # 紧固件验证
    fastener_keywords = ['螺丝', '螺钉', '螺栓', '螺母', '垫圈', 'screw', 'bolt', 'nut', 'washer']
    fasteners = df[df['drawing_name'].str.contains('|'.join(fastener_keywords), case=False, na=False)]

    print(f"  紧固件验证:")
    print(f"    关键词匹配: {len(fasteners)} 个图纸")
    print(f"    实际分类为紧固件: {len(df[df['product_category'].isin(['螺栓螺钉', '螺母', '垫圈', '六角螺栓', '垫片'])])} 个图纸")

    # 检查错误分类
    fastener_errors = fasteners[~fasteners['product_category'].isin(['螺栓螺钉', '螺母', '垫圈', '六角螺栓', '垫片', '未分类'])]
    if len(fastener_errors) > 0:
        print(f"    ❌ 可能的错误分类:")
        for _, row in fastener_errors.iterrows():
            print(f"      {row['drawing_name'][:50]}... → {row['product_category']}")
    else:
        print(f"    ✅ 紧固件分类正确")
    print()

    # 4. 未分类图纸分析
    unclassified = df[df['product_category'] == '未分类']
    print(f"📋 未分类图纸分析 (前10个):")
    for _, row in unclassified.head(10).iterrows():
        drawing_name = row['drawing_name']
        print(f"  {drawing_name[:50]}...")

        # 简单的分类建议
        lower_name = drawing_name.lower()
        if any(kw in lower_name for kw in ['screw', 'bolt', 'nut', 'washer']):
            print(f"    💡 建议: 紧固件")
        elif any(kw in lower_name for kw in ['钢', 'steel', 'metal']):
            print(f"    💡 建议: 建材-金属材料")
        elif any(kw in lower_name for kw in ['chair', 'table', 'cabinet']):
            print(f"    💡 建议: 家具")
        else:
            print(f"    💡 建议: 保持未分类")
    print()

    # 5. 分类质量评分
    print("📊 分类质量评分:")

    # 计算分类覆盖率
    coverage_rate = classified_drawings / total_drawings
    print(f"  分类覆盖率: {coverage_rate*100:.1f}%")

    # 计算紧固件识别准确率 (基于关键词)
    fastner_classified = df[df['product_category'].isin(['螺栓螺钉', '螺母', '垫圈', '六角螺栓', '垫片'])]
    if len(fasteners) > 0:
        accuracy = len(fasteners[fasteners['product_category'].isin(['螺栓螺钉', '螺母', '垫圈', '六角螺栓', '垫片'])]) / len(fasteners)
        print(f"  紧固件识别准确率: {accuracy*100:.1f}%")

    # 总体评分
    if coverage_rate > 0.1:
        grade = "A"
    elif coverage_rate > 0.05:
        grade = "B"
    else:
        grade = "C"

    print(f"  总体分类等级: {grade}")
    print()

    # 6. 改进建议
    print("💡 改进建议:")
    if coverage_rate < 0.2:
        print("  - 扩展关键词库以提高分类覆盖率")
        print("  - 添加更多产品类别的识别规则")

    if len(fastener_errors) > 0:
        print("  - 优化紧固件分类逻辑")

    print("  - 定期更新分类规则")
    print("  - 添加人工审核流程")

    conn.close()

if __name__ == "__main__":
    verify_classification()