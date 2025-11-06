#!/usr/bin/env python3
"""
数据模型扩充脚本
为现有数据库表添加新字段和索引
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('ModelUpdater')

def backup_database(db_path: str, logger) -> str:
    """备份数据库"""
    import shutil
    from datetime import datetime

    backup_dir = Path("./data/backups")
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"db_before_model_update_{timestamp}.sqlite"

    try:
        shutil.copy2(db_path, backup_file)
        logger.info(f"✅ 数据库备份完成: {backup_file}")
        return str(backup_file)
    except Exception as e:
        logger.error(f"❌ 数据库备份失败: {e}")
        return ""

def update_models(db_path: str = "./data/db.sqlite"):
    """更新数据模型"""
    logger = setup_logging()
    logger.info("🚀 开始更新数据模型...")

    # 备份数据库
    backup_file = backup_database(db_path, logger)
    if not backup_file:
        logger.error("❌ 备份失败，终止更新")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 开启事务
        cursor.execute("BEGIN TRANSACTION")

        # 1. 更新 Drawings 表
        logger.info("📝 更新 Drawings 表...")

        # 检查字段是否已存在
        cursor.execute("PRAGMA table_info(drawings)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # 添加新字段
        new_drawing_fields = {
            'standard_or_custom': 'BOOLEAN DEFAULT 0',
            'data_source': 'TEXT DEFAULT "manual"',
            'is_classified': 'BOOLEAN DEFAULT 0',
            'classification_confidence': 'REAL DEFAULT 0.0',
            'classification_date': 'TEXT'
        }

        for field, field_def in new_drawing_fields.items():
            if field not in existing_columns:
                alter_sql = f"ALTER TABLE drawings ADD COLUMN {field} {field_def}"
                cursor.execute(alter_sql)
                logger.info(f"  ✅ 添加字段: drawings.{field}")
            else:
                logger.info(f"  ⚠️ 字段已存在: drawings.{field}")

        # 2. 更新 Customers 表
        logger.info("📝 更新 Customers 表...")

        cursor.execute("PRAGMA table_info(customers)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # 添加新字段
        new_customer_fields = {
            'customer_status': 'TEXT DEFAULT "potential"',
            'last_inquiry_date': 'TEXT',
            'customer_level': 'TEXT DEFAULT "normal"',
            'total_drawings': 'INTEGER DEFAULT 0',
            'first_contact_date': 'TEXT',
            'contact_frequency': 'INTEGER DEFAULT 0',
            'notes': 'TEXT'
        }

        for field, field_def in new_customer_fields.items():
            if field not in existing_columns:
                alter_sql = f"ALTER TABLE customers ADD COLUMN {field} {field_def}"
                cursor.execute(alter_sql)
                logger.info(f"  ✅ 添加字段: customers.{field}")
            else:
                logger.info(f"  ⚠️ 字段已存在: customers.{field}")

        # 3. 更新 FactoryQuote 表
        logger.info("📝 更新 FactoryQuote 表...")

        cursor.execute("PRAGMA table_info(factory_quotes)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # 添加新字段
        new_quote_fields = {
            'quote_month': 'TEXT',
            'quote_quarter': 'TEXT',
            'quote_year': 'INTEGER',
            'price_change_pct': 'REAL',
            'is_standard_pricing': 'BOOLEAN DEFAULT 0',
            'quote_source': 'TEXT DEFAULT "manual"',
            'valid_until': 'TEXT'
        }

        for field, field_def in new_quote_fields.items():
            if field not in existing_columns:
                alter_sql = f"ALTER TABLE factory_quotes ADD COLUMN {field} {field_def}"
                cursor.execute(alter_sql)
                logger.info(f"  ✅ 添加字段: factory_quotes.{field}")
            else:
                logger.info(f"  ⚠️ 字段已存在: factory_quotes.{field}")

        # 4. 更新 Specifications 表
        logger.info("📝 更新 Specifications 表...")

        cursor.execute("PRAGMA table_info(specifications)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # 添加新字段
        new_spec_fields = {
            'spec_source': 'TEXT DEFAULT "manual"',
            'last_updated': 'TEXT',
            'is_active': 'BOOLEAN DEFAULT 1',
            'spec_version': 'TEXT DEFAULT "1.0"',
            'supplier_id': 'INTEGER'
        }

        for field, field_def in new_spec_fields.items():
            if field not in existing_columns:
                alter_sql = f"ALTER TABLE specifications ADD COLUMN {field} {field_def}"
                cursor.execute(alter_sql)
                logger.info(f"  ✅ 添加字段: specifications.{field}")
            else:
                logger.info(f"  ⚠️ 字段已存在: specifications.{field}")

        # 5. 添加新的索引
        logger.info("📊 创建新索引...")

        new_indexes = [
            # Drawings 表索引
            "CREATE INDEX IF NOT EXISTS idx_drawings_standard_custom ON drawings(standard_or_custom)",
            "CREATE INDEX IF NOT EXISTS idx_drawings_data_source ON drawings(data_source)",
            "CREATE INDEX IF NOT EXISTS idx_drawings_is_classified ON drawings(is_classified)",
            "CREATE INDEX IF NOT EXISTS idx_drawings_classification_confidence ON drawings(classification_confidence)",
            "CREATE INDEX IF NOT EXISTS idx_drawings_classification_date ON drawings(classification_date)",
            "CREATE INDEX IF NOT EXISTS idx_drawings_category_classified ON drawings(product_category, is_classified)",

            # Customers 表索引
            "CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(customer_status)",
            "CREATE INDEX IF NOT EXISTS idx_customers_last_inquiry ON customers(last_inquiry_date)",
            "CREATE INDEX IF NOT EXISTS idx_customers_level ON customers(customer_level)",
            "CREATE INDEX IF NOT EXISTS idx_customers_total_drawings ON customers(total_drawings)",
            "CREATE INDEX IF NOT EXISTS idx_customers_first_contact ON customers(first_contact_date)",
            "CREATE INDEX IF NOT EXISTS idx_customers_contact_frequency ON customers(contact_frequency)",
            "CREATE INDEX IF NOT EXISTS idx_customers_status_level ON customers(customer_status, customer_level)",

            # FactoryQuote 表索引
            "CREATE INDEX IF NOT EXISTS idx_quotes_quote_month ON factory_quotes(quote_month)",
            "CREATE INDEX IF NOT EXISTS idx_quotes_quote_quarter ON factory_quotes(quote_quarter)",
            "CREATE INDEX IF NOT EXISTS idx_quotes_quote_year ON factory_quotes(quote_year)",
            "CREATE INDEX IF NOT EXISTS idx_quotes_price_change ON factory_quotes(price_change_pct)",
            "CREATE INDEX IF NOT EXISTS idx_quotes_is_standard_pricing ON factory_quotes(is_standard_pricing)",
            "CREATE INDEX IF NOT EXISTS idx_quotes_quote_source ON factory_quotes(quote_source)",
            "CREATE INDEX IF NOT EXISTS idx_quotes_valid_until ON factory_quotes(valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_quotes_factory_month_category ON factory_quotes(factory_id, quote_month, product_category)",

            # Specifications 表索引
            "CREATE INDEX IF NOT EXISTS idx_specifications_source ON specifications(spec_source)",
            "CREATE INDEX IF NOT EXISTS idx_specifications_last_updated ON specifications(last_updated)",
            "CREATE INDEX IF NOT EXISTS idx_specifications_is_active ON specifications(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_specifications_version ON specifications(spec_version)",
            "CREATE INDEX IF NOT EXISTS idx_specifications_supplier ON specifications(supplier_id)"
        ]

        for index_sql in new_indexes:
            cursor.execute(index_sql)
            logger.info(f"  ✅ 创建索引: {index_sql.split('idx_')[1].split(' ')[0]}")

        # 6. 初始化数据
        logger.info("🔄 初始化新字段数据...")

        # 初始化客户统计数据
        cursor.execute("""
            UPDATE customers
            SET total_drawings = (
                SELECT COUNT(*)
                FROM drawings
                WHERE drawings.customer_id = customers.id
            )
        """)

        # 初始化报价时间字段
        cursor.execute("""
            UPDATE factory_quotes
            SET quote_month = substr(quote_date, 1, 7),
                quote_year = CAST(substr(quote_date, 1, 4) AS INTEGER),
                quote_quarter = CASE
                    WHEN CAST(substr(quote_date, 6, 2) AS INTEGER) IN (1,2,3) THEN 'Q1'
                    WHEN CAST(substr(quote_date, 6, 2) AS INTEGER) IN (4,5,6) THEN 'Q2'
                    WHEN CAST(substr(quote_date, 6, 2) AS INTEGER) IN (7,8,9) THEN 'Q3'
                    ELSE 'Q4'
                END
            WHERE quote_date IS NOT NULL
        """)

        # 提交事务
        conn.commit()
        logger.info("✅ 数据模型更新完成")

        # 7. 验证更新结果
        logger.info("🔍 验证更新结果...")

        # 检查表结构
        for table_name in ['customers', 'drawings', 'factory_quotes', 'specifications']:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            logger.info(f"  📋 {table_name}: {len(columns)} 个字段")

        # 检查索引数量
        cursor.execute("""
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        index_count = cursor.fetchone()[0]
        logger.info(f"  📊 总索引数: {index_count}")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"❌ 数据模型更新失败: {e}")
        logger.error(f"💡 数据库备份位于: {backup_file}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def generate_migration_report(db_path: str):
    """生成迁移报告"""
    logger = setup_logging()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 收集表结构信息
        tables_info = {}
        for table_name in ['customers', 'drawings', 'factory_quotes', 'specifications']:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            tables_info[table_name] = [
                {
                    'name': col[1],
                    'type': col[2],
                    'not_null': bool(col[3]),
                    'default': col[4],
                    'primary_key': bool(col[5])
                }
                for col in columns
            ]

        # 收集索引信息
        cursor.execute("""
            SELECT name, tbl_name, sql
            FROM sqlite_master
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
            ORDER BY tbl_name, name
        """)
        indexes = cursor.fetchall()

        # 生成报告
        report = {
            'migration_date': datetime.now().isoformat(),
            'database_path': db_path,
            'tables': tables_info,
            'indexes': [
                {
                    'name': idx[0],
                    'table': idx[1],
                    'sql': idx[2]
                }
                for idx in indexes
            ],
            'summary': {
                'total_tables': len(tables_info),
                'total_indexes': len(indexes)
            }
        }

        # 保存报告
        report_file = f"./data/processed/model_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("./data/processed").mkdir(exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"📄 迁移报告已保存: {report_file}")
        conn.close()
        return report_file

    except Exception as e:
        logger.error(f"❌ 生成迁移报告失败: {e}")
        return ""

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='数据模型更新工具')
    parser.add_argument('--db-path', default='./data/db.sqlite', help='数据库文件路径')
    parser.add_argument('--backup-only', action='store_true', help='仅备份数据库')
    parser.add_argument('--report-only', action='store_true', help='仅生成报告')

    args = parser.parse_args()

    if args.backup_only:
        logger = setup_logging()
        backup_file = backup_database(args.db_path, logger)
        if backup_file:
            print(f"✅ 备份完成: {backup_file}")
        else:
            print("❌ 备份失败")
    elif args.report_only:
        report_file = generate_migration_report(args.db_path)
        if report_file:
            print(f"✅ 报告生成完成: {report_file}")
        else:
            print("❌ 报告生成失败")
    else:
        success = update_models(args.db_path)
        if success:
            report_file = generate_migration_report(args.db_path)
            print(f"✅ 数据模型更新完成！")
            if report_file:
                print(f"📄 详细报告: {report_file}")
        else:
            print("❌ 数据模型更新失败，请检查日志")

if __name__ == "__main__":
    main()