#!/usr/bin/env python3
"""
数据库优化脚本
为关键字段添加索引，优化查询性能
"""

import sqlite3
from datetime import datetime
from models import DatabaseManager

class DatabaseOptimizer:
    """数据库优化器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)

    def optimize_database(self):
        """优化数据库性能"""
        print("🚀 数据库优化开始...")
        print(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 检查现有索引
            print("📊 检查现有索引...")
            self._check_existing_indexes(cursor)

            # 添加客户表索引
            print("\n👥 优化客户表索引...")
            self._add_customer_indexes(cursor)

            # 添加图纸表索引
            print("\n📄 优化图纸表索引...")
            self._add_drawing_indexes(cursor)

            # 添加其他表索引
            print("\n🏭 优化其他表索引...")
            self._add_other_indexes(cursor)

            # 创建唯一约束
            print("\n🔒 添加唯一约束...")
            self._add_unique_constraints(cursor)

            # 分析表统计信息
            print("\n📈 分析表统计信息...")
            self._analyze_tables(cursor)

            # 验证优化效果
            print("\n✅ 验证优化效果...")
            self._verify_optimization(cursor)

            conn.commit()

        print("\n🎉 数据库优化完成!")

    def _check_existing_indexes(self, cursor):
        """检查现有索引"""
        cursor.execute("""
            SELECT name, tbl_name FROM sqlite_master
            WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            ORDER BY tbl_name, name
        """)

        indexes = cursor.fetchall()
        print(f"现有索引数量: {len(indexes)}")

        if indexes:
            print("现有索引列表:")
            for name, table in indexes:
                print(f"  - {table}.{name}")

    def _add_customer_indexes(self, cursor):
        """添加客户表索引"""
        indexes_to_add = [
            # 基础查询索引
            ("idx_customers_company_name", "customers", "company_name"),
            ("idx_customers_contact_email", "customers", "contact_email"),
            ("idx_customers_country", "customers", "country"),
            ("idx_customers_language", "customers", "language"),

            # 复合索引 - 支持核心查询
            ("idx_customers_company_email", "customers", "company_name, contact_email"),
            ("idx_customers_name_email_country", "customers", "company_name, contact_email, country"),

            # 时间索引
            ("idx_customers_first_contact", "customers", "first_contact_date"),
            ("idx_customers_created_at", "customers", "created_at"),
            ("idx_customers_updated_at", "customers", "updated_at")
        ]

        for index_name, table, columns in indexes_to_add:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({columns})")
                print(f"  ✅ 创建索引: {index_name}")
            except sqlite3.Error as e:
                print(f"  ❌ 创建索引失败 {index_name}: {e}")

    def _add_drawing_indexes(self, cursor):
        """添加图纸表索引"""
        indexes_to_add = [
            # 基础查询索引
            ("idx_drawings_customer_id", "drawings", "customer_id"),
            ("idx_drawings_product_category", "drawings", "product_category"),
            ("idx_drawings_status", "drawings", "status"),
            ("idx_drawings_upload_date", "drawings", "upload_date"),

            # 文件相关索引
            ("idx_drawings_file_path", "drawings", "file_path"),
            ("idx_drawings_drawing_name", "drawings", "drawing_name"),

            # 复合索引 - 支持常用查询
            ("idx_drawings_customer_status", "drawings", "customer_id, status"),
            ("idx_drawings_category_status", "drawings", "product_category, status"),
            ("idx_drawings_customer_category", "drawings", "customer_id, product_category"),

            # 时间索引
            ("idx_drawings_created_at", "drawings", "created_at"),
            ("idx_drawings_updated_at", "drawings", "updated_at")
        ]

        for index_name, table, columns in indexes_to_add:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({columns})")
                print(f"  ✅ 创建索引: {index_name}")
            except sqlite3.Error as e:
                print(f"  ❌ 创建索引失败 {index_name}: {e}")

    def _add_other_indexes(self, cursor):
        """添加其他表索引"""
        indexes_to_add = [
            # 工厂表索引
            ("idx_factories_name", "factories", "factory_name"),
            ("idx_factories_location", "factories", "location"),
            ("idx_factories_capability", "factories", "capability"),

            # 工厂报价表索引
            ("idx_factory_quotes_factory_id", "factory_quotes", "factory_id"),
            ("idx_factory_quotes_category", "factory_quotes", "product_category"),
            ("idx_factory_quotes_quote_date", "factory_quotes", "quote_date"),
            ("idx_factory_quotes_factory_category", "factory_quotes", "factory_id, product_category"),

            # 规格表索引
            ("idx_specifications_category", "specifications", "product_category"),
            ("idx_specifications_material", "specifications", "material"),
            ("idx_specifications_standard_custom", "specifications", "standard_or_custom"),
            ("idx_specifications_category_material", "specifications", "product_category, material"),

            # 流程状态表索引
            ("idx_process_status_drawing_id", "process_status", "drawing_id"),
            ("idx_process_status_customer_id", "process_status", "customer_id"),
            ("idx_process_status_status", "process_status", "status"),
            ("idx_process_status_last_update", "process_status", "last_update_date"),
            ("idx_process_status_customer_status", "process_status", "customer_id, status")
        ]

        for index_name, table, columns in indexes_to_add:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({columns})")
                print(f"  ✅ 创建索引: {index_name}")
            except sqlite3.Error as e:
                print(f"  ❌ 创建索引失败 {index_name}: {e}")

    def _add_unique_constraints(self, cursor):
        """添加唯一约束"""
        constraints_to_add = [
            # 为contact_email添加唯一约束（允许NULL）
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_customers_email
            ON customers(contact_email)
            WHERE contact_email IS NOT NULL AND contact_email != ''
            """,

            # 为公司名称+邮箱组合添加唯一约束
            ("uq_customers_company_email", "customers", "company_name, contact_email",
             "WHERE contact_email IS NOT NULL AND contact_email != ''")
        ]

        for constraint in constraints_to_add:
            try:
                if isinstance(constraint, tuple):
                    name, table, columns, where = constraint
                    sql = f"CREATE UNIQUE INDEX IF NOT EXISTS {name} ON {table} ({columns}) {where}"
                else:
                    sql = constraint

                cursor.execute(sql)
                print(f"  ✅ 创建唯一约束")
            except sqlite3.Error as e:
                print(f"  ❌ 创建唯一约束失败: {e}")

    def _analyze_tables(self, cursor):
        """分析表统计信息"""
        tables = ['customers', 'factories', 'drawings', 'factory_quotes', 'specifications', 'process_status']

        for table in tables:
            try:
                cursor.execute(f"ANALYZE {table}")
                print(f"  ✅ 分析表: {table}")
            except sqlite3.Error as e:
                print(f"  ❌ 分析表失败 {table}: {e}")

    def _verify_optimization(self, cursor):
        """验证优化效果"""
        # 检查总索引数
        cursor.execute("""
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
        """)
        total_indexes = cursor.fetchone()[0]
        print(f"  📊 总索引数: {total_indexes}")

        # 测试查询性能（简单示例）
        queries_to_test = [
            ("按公司名查询客户", "SELECT COUNT(*) FROM customers WHERE company_name LIKE '%AYA%'"),
            ("按邮箱查询客户", "SELECT COUNT(*) FROM customers WHERE contact_email = 'info@aya-fasteners.com'"),
            ("按客户查询图纸", "SELECT COUNT(*) FROM drawings WHERE customer_id = 4"),
            ("按类别查询图纸", "SELECT COUNT(*) FROM drawings WHERE product_category = 'screw'"),
            ("按状态查询图纸", "SELECT COUNT(*) FROM drawings WHERE status = 'pending'")
        ]

        print("  🔍 测试查询性能:")
        for desc, query in queries_to_test:
            try:
                start_time = datetime.now()
                cursor.execute(query)
                result = cursor.fetchone()
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() * 1000

                print(f"    {desc}: {result[0]} 条记录 ({duration:.2f}ms)")
            except sqlite3.Error as e:
                print(f"    {desc}: 查询失败 - {e}")

    def show_index_usage(self):
        """显示索引使用情况（需要SQLite特定版本支持）"""
        print("\n📊 索引使用情况:")
        try:
            with self.db_manager:
                conn = self.db_manager.connect()
                cursor = conn.cursor()

                # 这个查询在某些SQLite版本中可能不支持
                cursor.execute("PRAGMA index_list(customers)")
                indexes = cursor.fetchall()

                for index in indexes:
                    print(f"  - {index[1]} (unique: {index[2]})")

        except sqlite3.Error:
            print("  ⚠️  当前SQLite版本不支持索引使用情况查询")

def main():
    """主函数"""
    optimizer = DatabaseOptimizer()
    optimizer.optimize_database()

    # 显示索引使用情况
    optimizer.show_index_usage()

if __name__ == "__main__":
    main()