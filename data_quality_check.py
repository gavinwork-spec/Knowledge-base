#!/usr/bin/env python3
"""
数据质量检查脚本
验证客户数据的完整性和唯一性
"""

import sqlite3
import pandas as pd
import re
from collections import Counter

def check_email_quality():
    """检查邮箱质量"""
    db_path = "./data/db.sqlite"
    conn = sqlite3.connect(db_path)

    # 获取所有客户数据
    df = pd.read_sql_query("SELECT * FROM customers", conn)

    print("🔍 数据质量检查报告")
    print("=" * 50)

    # 1. 基本统计
    print(f"📊 客户数据统计:")
    print(f"  总客户数: {len(df)}")
    print(f"  有联系邮箱: {len(df[df['contact_email'].notna()])}")
    print(f"  缺失邮箱: {len(df[df['contact_email'].isna()])}")
    print()

    # 2. 邮箱格式验证
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    invalid_emails = []
    for idx, row in df.iterrows():
        email = row['contact_email']
        if pd.isna(email):
            continue

        if not re.match(email_pattern, str(email)):
            invalid_emails.append({
                'company_name': row['company_name'],
                'contact_email': email,
                'issue': '格式错误'
            })

    print(f"📧 邮箱格式检查:")
    print(f"  有效邮箱: {len(df[df['contact_email'].notna()]) - len(invalid_emails)}")
    print(f"  无效邮箱: {len(invalid_emails)}")

    if invalid_emails:
        print(f"  ❌ 无效邮箱列表:")
        for item in invalid_emails[:5]:  # 只显示前5个
            print(f"    {item['company_name']}: {item['contact_email']}")
    print()

    # 3. 唯一性检查
    # 检查邮箱重复
    email_counts = df['contact_email'].value_counts()
    duplicate_emails = email_counts[email_counts > 1]

    print(f"🔄 邮箱唯一性检查:")
    print(f"  唯一邮箱: {len(email_counts[email_counts == 1])}")
    print(f"  重复邮箱: {len(duplicate_emails)}")

    if len(duplicate_emails) > 0:
        print(f"  ❌ 重复邮箱列表:")
        for email, count in duplicate_emails.head().items():
            print(f"    {email}: {count} 次")
    print()

    # 4. 公司名称检查
    # 检查空公司名称
    empty_company = df[df['company_name'].isna() | (df['company_name'] == '')]

    # 检查公司名称重复
    company_counts = df['company_name'].value_counts()
    duplicate_companies = company_counts[company_counts > 1]

    print(f"🏢 公司名称检查:")
    print(f"  有效公司名: {len(df) - len(empty_company)}")
    print(f"  缺失公司名: {len(empty_company)}")
    print(f"  重复公司名: {len(duplicate_companies)}")

    if len(empty_company) > 0:
        print(f"  ❌ 缺失公司名的客户:")
        for idx, row in empty_company.iterrows():
            print(f"    ID {row['id']}: {row['contact_email']}")
    print()

    # 5. 组合唯一性检查 (contact_email + company_name)
    # 这是核心客户标识
    df['unique_key'] = df['company_name'].fillna('') + '|' + df['contact_email'].fillna('')
    key_counts = df['unique_key'].value_counts()
    duplicate_keys = key_counts[key_counts > 1]

    print(f"🔑 核心标识检查 (company_name + contact_email):")
    print(f"  唯一标识: {len(key_counts[key_counts == 1])}")
    print(f"  重复标识: {len(duplicate_keys)}")

    if len(duplicate_keys) > 0:
        print(f"  ❌ 重复的客户标识:")
        for key, count in duplicate_keys.head().items():
            company, email = key.split('|')
            print(f"    {company} + {email}: {count} 次")
    print()

    # 6. 数据完整性评分
    print(f"📊 数据质量评分:")

    # 计算各项指标得分
    email_completeness = len(df[df['contact_email'].notna()]) / len(df)
    email_validity = (len(df[df['contact_email'].notna()]) - len(invalid_emails)) / len(df)
    email_uniqueness = len(email_counts[email_counts == 1]) / len(df)
    company_completeness = len(df[df['company_name'].notna() & (df['company_name'] != '')]) / len(df)
    key_uniqueness = len(key_counts[key_counts == 1]) / len(df)

    scores = {
        '邮箱完整性': email_completeness * 100,
        '邮箱有效性': email_validity * 100,
        '邮箱唯一性': email_uniqueness * 100,
        '公司名称完整性': company_completeness * 100,
        '核心标识唯一性': key_uniqueness * 100
    }

    for metric, score in scores.items():
        status = "✅" if score >= 90 else "⚠️" if score >= 70 else "❌"
        print(f"  {status} {metric}: {score:.1f}%")

    # 总体评分
    overall_score = sum(scores.values()) / len(scores)
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B"
    elif overall_score >= 70:
        grade = "C"
    else:
        grade = "D"

    print(f"\n🎯 总体数据质量等级: {grade} ({overall_score:.1f}%)")

    # 7. 修复建议
    print(f"\n💡 数据修复建议:")

    if len(invalid_emails) > 0:
        print(f"  - 修复 {len(invalid_emails)} 个无效邮箱格式")

    if len(duplicate_emails) > 0:
        print(f"  - 处理 {len(duplicate_emails)} 个重复邮箱")

    if len(empty_company) > 0:
        print(f"  - 补充 {len(empty_company)} 个缺失的公司名称")

    if len(duplicate_companies) > 0:
        print(f"  - 检查 {len(duplicate_companies)} 个重复的公司名称")

    if len(duplicate_keys) > 0:
        print(f"  - 清理 {len(duplicate_keys)} 个重复的客户记录")

    conn.close()

def check_data_relationships():
    """检查数据关系完整性"""
    db_path = "./data/db.sqlite"
    conn = sqlite3.connect(db_path)

    print(f"\n🔗 数据关系检查:")
    print("-" * 30)

    # 检查客户-图纸关系
    orphan_drawings = pd.read_sql_query("""
        SELECT d.id, d.drawing_name, d.customer_id
        FROM drawings d
        LEFT JOIN customers c ON d.customer_id = c.id
        WHERE c.id IS NULL
    """, conn)

    print(f"客户-图纸关系:")
    print(f"  孤立图纸: {len(orphan_drawings)} 个")

    if len(orphan_drawings) > 0:
        print(f"  ❌ 孤立图纸列表 (前5个):")
        for _, row in orphan_drawings.head().iterrows():
            print(f"    ID {row['id']}: {row['drawing_name'][:30]}... (customer_id: {row['customer_id']})")

    # 检查工厂-报价关系 (factory_quotes表通过factory_id关联factories)
    orphan_quotes = pd.read_sql_query("""
        SELECT q.id, q.quote_date, q.factory_id
        FROM factory_quotes q
        LEFT JOIN factories f ON q.factory_id = f.id
        WHERE f.id IS NULL
    """, conn)

    print(f"工厂-报价关系:")
    print(f"  孤立报价: {len(orphan_quotes)} 个")

    if len(orphan_quotes) > 0:
        print(f"  ❌ 孤立报价列表:")
        for _, row in orphan_quotes.iterrows():
            print(f"    ID {row['id']}: {row['quote_date']} (factory_id: {row['factory_id']})")

    # specifications表没有drawing_id字段，检查产品类别关联
    print(f"规格数据:")
    spec_count = pd.read_sql_query("SELECT COUNT(*) as count FROM specifications", conn)
    print(f"  总规格数: {spec_count.iloc[0]['count']} 个")

    conn.close()

if __name__ == "__main__":
    check_email_quality()
    check_data_relationships()