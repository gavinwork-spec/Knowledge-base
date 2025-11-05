#!/usr/bin/env python3
"""
数据库验证脚本
检查数据库中的数据状态，提供详细的统计信息
"""

import sqlite3
from datetime import datetime
from models import DatabaseManager

def verify_database():
    """验证数据库内容"""
    print("=" * 60)
    print("数据库验证报告")
    print("=" * 60)

    db_manager = DatabaseManager("./data/db.sqlite")

    with db_manager:
        conn = db_manager.connect()
        cursor = conn.cursor()

        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        print(f"📊 数据库表总数: {len(tables)}")
        print()

        # 检查每个表的数据
        table_stats = {}

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            table_stats[table] = count
            print(f"📋 {table}: {count} 条记录")

        print("\n" + "=" * 60)
        print("详细数据分析")
        print("=" * 60)

        # Customer 表详细分析
        if table_stats.get('customers', 0) > 0:
            print("\n👥 客户分析:")
            cursor.execute("""
                SELECT
                    company_name,
                    contact_email,
                    country,
                    COUNT(*) as count
                FROM customers
                GROUP BY company_name, contact_email
                ORDER BY count DESC
                LIMIT 10
            """)

            customers = cursor.fetchall()
            for company, email, country, count in customers:
                print(f"  • {company} ({email}) - {country}")

        # Drawing 表详细分析
        if table_stats.get('drawings', 0) > 0:
            print("\n📄 图纸分析:")
            cursor.execute("""
                SELECT
                    product_category,
                    status,
                    COUNT(*) as count
                FROM drawings
                GROUP BY product_category, status
                ORDER BY count DESC
                LIMIT 10
            """)

            drawings = cursor.fetchall()
            for category, status, count in drawings:
                print(f"  • {category} ({status}) - {count} 个文件")

            # 显示关联客户的图纸
            cursor.execute("""
                SELECT COUNT(*) as with_customer,
                       COUNT(*) - COUNT(customer_id) as without_customer
                FROM drawings
            """)
            result = cursor.fetchone()
            print(f"  🔗 已关联客户的图纸: {result[0]} 个")
            print(f"  ❓ 未关联客户的图纸: {result[1]} 个")

        # ProcessStatus 表分析
        if table_stats.get('process_status', 0) > 0:
            print("\n🔄 流程状态分析:")
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM process_status
                GROUP BY status
                ORDER BY count DESC
            """)

            statuses = cursor.fetchall()
            for status, count in statuses:
                print(f"  • {status}: {count} 条记录")

        # 检查最近的处理日志
        print("\n📝 最近处理日志:")
        try:
            import json
            from pathlib import Path

            # 检查客户处理日志
            customer_log = Path("./data/processed/customer_ingest_log.json")
            if customer_log.exists():
                with open(customer_log, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                print(f"  客户导入日志:")
                print(f"    扫描时间: {log_data.get('scan_time', 'Unknown')}")
                print(f"    处理记录: {log_data.get('processed_count', 0)}")
                print(f"    错误数量: {log_data.get('error_count', 0)}")

            # 检查图纸处理日志
            drawing_log = Path("./data/processed/drawing_ingest_log.json")
            if drawing_log.exists():
                with open(drawing_log, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                print(f"  图纸导入日志:")
                print(f"    扫描时间: {log_data.get('scan_time', 'Unknown')}")
                print(f"    处理记录: {log_data.get('processed_count', 0)}")
                print(f"    错误数量: {log_data.get('error_count', 0)}")

        except Exception as e:
            print(f"  ⚠️  读取日志失败: {e}")

        # 数据质量检查
        print("\n🔍 数据质量检查:")

        # 检查重复客户
        if table_stats.get('customers', 0) > 0:
            cursor.execute("""
                SELECT company_name, contact_email, COUNT(*) as count
                FROM customers
                GROUP BY company_name, contact_email
                HAVING count > 1
            """)

            duplicates = cursor.fetchall()
            if duplicates:
                print(f"  ⚠️  发现 {len(duplicates)} 组重复客户:")
                for company, email, count in duplicates[:5]:  # 只显示前5个
                    print(f"    • {company} / {email} - {count} 条记录")
            else:
                print("  ✅ 未发现重复客户记录")

        # 检查孤儿记录（外键引用不存在）
        if table_stats.get('drawings', 0) > 0:
            cursor.execute("""
                SELECT COUNT(*)
                FROM drawings d
                LEFT JOIN customers c ON d.customer_id = c.id
                WHERE d.customer_id IS NOT NULL AND c.id IS NULL
            """)

            orphan_drawings = cursor.fetchone()[0]
            if orphan_drawings > 0:
                print(f"  ⚠️  发现 {orphan_drawings} 个图纸记录引用了不存在的客户")
            else:
                print("  ✅ 所有图纸的客户引用都有效")

        # 总体统计
        print(f"\n📈 总体统计:")
        total_records = sum(table_stats.values())
        print(f"  • 总记录数: {total_records}")
        print(f"  • 有数据的表: {sum(1 for count in table_stats.values() if count > 0)}/{len(tables)}")

        print("\n" + "=" * 60)
        print("验证完成!")
        print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

if __name__ == "__main__":
    verify_database()