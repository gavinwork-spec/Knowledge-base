#!/usr/bin/env python3
"""
知识库管理脚本
提供数据库管理、批量导入、统计报告等功能
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from models import DatabaseManager, Customer, Drawing, Factory, FactoryQuote, Specification, ProcessStatus
from ingest_customers import CustomerIngestor
from ingest_drawings import DrawingIngestor
from verify_database import verify_database

class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)
        self.customer_ingestor = CustomerIngestor(db_path)
        self.drawing_ingestor = DrawingIngestor(db_path)

    def initialize_database(self):
        """初始化数据库"""
        print("🗄️ 正在初始化数据库...")
        from setup_models import main as setup_main
        setup_main()
        print("✅ 数据库初始化完成")

    def import_customers(self, directory: str = None):
        """导入客户资料"""
        if not directory:
            directory = "/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户/"

        print(f"👥 开始导入客户资料: {directory}")
        result = self.customer_ingestor.process_directory(directory)
        print(f"✅ 客户资料导入完成: {result}")
        return result

    def import_drawings(self, directory: str = None):
        """导入图纸资料"""
        if not directory:
            directory = "/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价/"

        print(f"📄 开始导入图纸资料: {directory}")
        result = self.drawing_ingestor.process_directory(directory)
        print(f"✅ 图纸资料导入完成: {result}")
        return result

    def full_import(self):
        """完整导入（客户+图纸）"""
        print("🚀 开始完整导入流程...")
        print("=" * 60)

        # 导入客户
        customer_result = self.import_customers()
        print()

        # 导入图纸
        drawing_result = self.import_drawings()
        print()

        # 验证数据库
        print("📊 生成验证报告...")
        verify_database()

        print("=" * 60)
        print("🎉 完整导入流程完成!")
        return {
            'customers': customer_result,
            'drawings': drawing_result
        }

    def show_statistics(self):
        """显示统计信息"""
        verify_database()

    def search_customers(self, keyword: str):
        """搜索客户"""
        customer = Customer(self.db_manager)
        all_customers = customer.get_all()

        print(f"🔍 搜索客户: {keyword}")
        matches = []

        for cust in all_customers:
            if (keyword.lower() in cust['company_name'].lower() or
                keyword.lower() in (cust.get('contact_email', '') or '').lower() or
                keyword.lower() in (cust.get('contact_name', '') or '').lower()):
                matches.append(cust)

        if matches:
            print(f"✅ 找到 {len(matches)} 个匹配的客户:")
            for i, cust in enumerate(matches, 1):
                print(f"  {i}. {cust['company_name']}")
                print(f"     联系人: {cust.get('contact_name', 'N/A')}")
                print(f"     邮箱: {cust.get('contact_email', 'N/A')}")
                print(f"     国家: {cust.get('country', 'N/A')}")
                print()
        else:
            print("❌ 未找到匹配的客户")

        return matches

    def search_drawings(self, keyword: str):
        """搜索图纸"""
        drawing = Drawing(self.db_manager)
        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM drawings
                WHERE drawing_name LIKE ? OR product_category LIKE ? OR notes LIKE ?
                ORDER BY upload_date DESC
                LIMIT 50
            """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))

            matches = [dict(row) for row in cursor.fetchall()]

        print(f"🔍 搜索图纸: {keyword}")
        if matches:
            print(f"✅ 找到 {len(matches)} 个匹配的图纸:")
            for i, draw in enumerate(matches, 1):
                print(f"  {i}. {draw['drawing_name']}")
                print(f"     类别: {draw['product_category']}")
                print(f"     状态: {draw['status']}")
                print(f"     路径: {draw['file_path']}")
                if draw['notes']:
                    print(f"     备注: {draw['notes'][:100]}...")
                print()
        else:
            print("❌ 未找到匹配的图纸")

        return matches

    def export_summary(self, output_file: str = None):
        """导出摘要报告"""
        if not output_file:
            output_file = f"./data/processed/knowledge_base_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        print(f"📋 导出摘要报告到: {output_file}")

        # 重定向输出到文件
        original_stdout = sys.stdout
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            print(f"知识库摘要报告")
            print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            verify_database()
        sys.stdout = original_stdout

        print(f"✅ 摘要报告已导出")

    def cleanup_temp_files(self):
        """清理临时文件"""
        print("🧹 清理临时文件...")
        temp_dirs = [
            "./data/processed",
            "/tmp/drawing_ingest_temp"
        ]

        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                # 清理超过7天的日志文件
                import time
                current_time = time.time()
                for file_path in Path(temp_dir).glob("*.log"):
                    if file_path.stat().st_mtime < current_time - 7*24*3600:
                        file_path.unlink()
                        print(f"  删除过期日志: {file_path}")

        print("✅ 临时文件清理完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识库管理工具")
    parser.add_argument('command', choices=[
        'init', 'import-customers', 'import-drawings', 'full-import',
        'stats', 'search-customers', 'search-drawings', 'export', 'cleanup'
    ], help='要执行的命令')

    parser.add_argument('--dir', help='指定目录路径')
    parser.add_argument('--keyword', help='搜索关键词')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--db', default='./data/db.sqlite', help='数据库文件路径')

    args = parser.parse_args()

    # 创建管理器
    manager = KnowledgeBaseManager(args.db)

    # 执行命令
    try:
        if args.command == 'init':
            manager.initialize_database()
        elif args.command == 'import-customers':
            manager.import_customers(args.dir)
        elif args.command == 'import-drawings':
            manager.import_drawings(args.dir)
        elif args.command == 'full-import':
            manager.full_import()
        elif args.command == 'stats':
            manager.show_statistics()
        elif args.command == 'search-customers':
            if not args.keyword:
                print("❌ 请提供搜索关键词 --keyword")
                return
            manager.search_customers(args.keyword)
        elif args.command == 'search-drawings':
            if not args.keyword:
                print("❌ 请提供搜索关键词 --keyword")
                return
            manager.search_drawings(args.keyword)
        elif args.command == 'export':
            manager.export_summary(args.output)
        elif args.command == 'cleanup':
            manager.cleanup_temp_files()

    except KeyboardInterrupt:
        print("\n⚠️ 操作被用户中断")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())