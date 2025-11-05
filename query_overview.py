#!/usr/bin/env python3
"""
知识库查询脚本
提供全面的客户、图纸、报价等查询功能
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from models import DatabaseManager, Customer, Drawing, Factory, FactoryQuote, Specification, ProcessStatus

class KnowledgeBaseQuery:
    """知识库查询器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)
        self.customer = Customer(self.db_manager)
        self.drawing = Drawing(self.db_manager)

    def run_interactive_mode(self):
        """运行交互式查询模式"""
        print("🔍 知识库查询系统")
        print("=" * 50)
        print("输入 'help' 查看可用命令，输入 'quit' 退出")
        print()

        while True:
            try:
                command = input("kb> ").strip().lower()

                if command in ['quit', 'exit', 'q']:
                    print("👋 再见!")
                    break
                elif command == 'help':
                    self._show_help()
                elif command == 'stats':
                    self.show_overview()
                elif command.startswith('customers'):
                    self._handle_customer_command(command)
                elif command.startswith('drawings'):
                    self._handle_drawing_command(command)
                elif command.startswith('search'):
                    self._handle_search_command(command)
                elif command.startswith('export'):
                    self._handle_export_command(command)
                else:
                    print(f"❌ 未知命令: {command}，输入 'help' 查看帮助")

            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 执行命令失败: {e}")

    def _show_help(self):
        """显示帮助信息"""
        print("""
可用命令:
  stats                    - 显示概览统计
  customers                - 列出所有客户
  customers --country CN    - 按国家筛选客户
  customers --detail ID     - 显示客户详细信息
  search --customer KEY     - 搜索客户
  search --drawing KEY      - 搜索图纸
  drawings                 - 列出所有图纸（概览）
  drawings --customer ID    - 按客户ID查询图纸
  drawings --category CAT   - 按类别查询图纸
  drawings --status STAT    - 按状态查询图纸
  export --customers FILE   - 导出客户列表
  export --drawings FILE    - 导出图纸列表
  help                     - 显示此帮助
  quit                     - 退出系统
        """)

    def show_overview(self):
        """显示知识库概览"""
        print("📊 知识库概览统计")
        print("=" * 50)

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 基础统计
            stats = {}
            tables = ['customers', 'factories', 'drawings', 'factory_quotes', 'specifications', 'process_status']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            print(f"👥 客户数量: {stats['customers']}")
            print(f"📄 图纸数量: {stats['drawings']}")
            print(f"🏭 工厂数量: {stats['factories']}")
            print(f"💰 报价数量: {stats['factory_quotes']}")
            print(f"📋 规格数量: {stats['specifications']}")
            print(f"🔄 流程状态: {stats['process_status']}")

            # 客户分布
            cursor.execute("SELECT country, COUNT(*) FROM customers GROUP BY country ORDER BY COUNT(*) DESC")
            countries = cursor.fetchall()
            print(f"\n🌍 客户地区分布:")
            for country, count in countries:
                print(f"  {country}: {count} 个客户")

            # 图纸分类
            cursor.execute("""
                SELECT product_category, COUNT(*)
                FROM drawings
                GROUP BY product_category
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)
            categories = cursor.fetchall()
            print(f"\n📂 图纸分类 (前10):")
            for category, count in categories:
                print(f"  {category}: {count} 个文件")

            # 客户关联率
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(customer_id) as linked
                FROM drawings
            """)
            total, linked = cursor.fetchone()
            if total > 0:
                linkage_rate = linked / total * 100
                print(f"\n🔗 图纸客户关联率: {linkage_rate:.1f}% ({linked}/{total})")

    def list_all_customers(self, country_filter=None):
        """列出所有客户"""
        print("👥 客户列表")
        print("=" * 50)

        customers = self.customer.get_all()

        if country_filter:
            customers = [c for c in customers if c.get('country') == country_filter]
            print(f"筛选国家: {country_filter}")

        if not customers:
            print("❌ 未找到客户")
            return

        print(f"共找到 {len(customers)} 个客户:\n")

        for i, customer in enumerate(customers, 1):
            print(f"{i:2d}. {customer['company_name']}")
            print(f"    联系人: {customer.get('contact_name', 'N/A')}")
            print(f"    邮箱: {customer.get('contact_email', 'N/A')}")
            print(f"    国家: {customer.get('country', 'N/A')}")
            print(f"    语言: {customer.get('language', 'N/A')}")
            print(f"    首次联系: {customer.get('first_contact_date', 'N/A')}")
            print()

    def show_customer_detail(self, customer_id):
        """显示客户详细信息"""
        print(f"👥 客户详细信息 (ID: {customer_id})")
        print("=" * 50)

        customer = self.customer.get_by_id(customer_id)
        if not customer:
            print(f"❌ 未找到客户 ID {customer_id}")
            return

        # 客户基本信息
        print(f"公司名称: {customer['company_name']}")
        print(f"联系人: {customer.get('contact_name', 'N/A')}")
        print(f"邮箱: {customer.get('contact_email', 'N/A')}")
        print(f"电话: {customer.get('phone', 'N/A')}")
        print(f"国家: {customer.get('country', 'N/A')}")
        print(f"语言: {customer.get('language', 'N/A')}")
        print(f"首次联系: {customer.get('first_contact_date', 'N/A')}")
        print(f"备注: {customer.get('notes', 'N/A')}")
        print(f"创建时间: {customer.get('created_at', 'N/A')}")

        # 关联的图纸
        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, drawing_name, product_category, status, upload_date
                FROM drawings
                WHERE customer_id = ?
                ORDER BY upload_date DESC
            """, (customer_id,))

            drawings = cursor.fetchall()

            if drawings:
                print(f"\n📄 关联图纸 ({len(drawings)} 个):")
                for drawing in drawings:
                    print(f"  • {drawing[1]} ({drawing[2]} - {drawing[3]})")
                    print(f"    上传时间: {drawing[4]}")
            else:
                print(f"\n📄 该客户暂无关联图纸")

    def search_customers(self, keyword):
        """搜索客户"""
        print(f"🔍 搜索客户: '{keyword}'")
        print("=" * 50)

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM customers
                WHERE company_name LIKE ?
                   OR contact_name LIKE ?
                   OR contact_email LIKE ?
                   OR notes LIKE ?
                ORDER BY company_name
            """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))

            customers = [dict(row) for row in cursor.fetchall()]

        if not customers:
            print("❌ 未找到匹配的客户")
            return

        print(f"找到 {len(customers)} 个匹配客户:\n")

        for i, customer in enumerate(customers, 1):
            print(f"{i}. {customer['company_name']}")
            print(f"   联系人: {customer.get('contact_name', 'N/A')}")
            print(f"   邮箱: {customer.get('contact_email', 'N/A')}")
            print(f"   国家: {customer.get('country', 'N/A')}")
            print()

    def search_drawings(self, keyword):
        """搜索图纸"""
        print(f"🔍 搜索图纸: '{keyword}'")
        print("=" * 50)

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT d.*, c.company_name
                FROM drawings d
                LEFT JOIN customers c ON d.customer_id = c.id
                WHERE d.drawing_name LIKE ?
                   OR d.product_category LIKE ?
                   OR d.notes LIKE ?
                ORDER BY d.upload_date DESC
                LIMIT 50
            """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))

            results = cursor.fetchall()

        if not results:
            print("❌ 未找到匹配的图纸")
            return

        print(f"找到 {len(results)} 个匹配图纸:\n")

        for i, row in enumerate(results, 1):
            print(f"{i}. {row[1]}")
            print(f"   类别: {row[3]}")
            print(f"   状态: {row[5]}")
            print(f"   客户: {row[10] or '未关联'}")
            print(f"   上传时间: {row[4]}")
            if row[7]:  # notes
                print(f"   备注: {row[7][:100]}...")
            print()

    def list_drawings_by_customer(self, customer_id=None):
        """按客户查询图纸"""
        if customer_id:
            customer = self.customer.get_by_id(customer_id)
            if not customer:
                print(f"❌ 未找到客户 ID {customer_id}")
                return
            print(f"📄 {customer['company_name']} 的图纸")
        else:
            print("📄 所有图纸（按客户分组）")

        print("=" * 50)

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            if customer_id:
                cursor.execute("""
                    SELECT drawing_name, product_category, status, upload_date, file_path
                    FROM drawings
                    WHERE customer_id = ?
                    ORDER BY upload_date DESC
                """, (customer_id,))
                results = cursor.fetchall()

                if not results:
                    print("该客户暂无图纸")
                    return

                for drawing in results:
                    print(f"• {drawing[0]}")
                    print(f"  类别: {drawing[1]}, 状态: {drawing[2]}")
                    print(f"  上传时间: {drawing[3]}")
                    print()

            else:
                cursor.execute("""
                    SELECT c.company_name, COUNT(*) as drawing_count
                    FROM customers c
                    LEFT JOIN drawings d ON c.id = d.customer_id
                    GROUP BY c.id, c.company_name
                    HAVING drawing_count > 0
                    ORDER BY drawing_count DESC
                """)

                customer_drawings = cursor.fetchall()

                if not customer_drawings:
                    print("❌ 暂无关联的图纸")
                    return

                print("客户图纸统计:\n")
                for company, count in customer_drawings:
                    print(f"• {company}: {count} 个图纸")

                # 显示未关联的图纸数量
                cursor.execute("SELECT COUNT(*) FROM drawings WHERE customer_id IS NULL")
                unlinked = cursor.fetchone()[0]
                if unlinked > 0:
                    print(f"\n未关联客户的图纸: {unlinked} 个")

    def list_drawings_by_category(self, category=None):
        """按类别查询图纸"""
        if category:
            print(f"📄 图纸类别: {category}")
        else:
            print("📄 图纸分类统计")

        print("=" * 50)

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            if category:
                cursor.execute("""
                    SELECT drawing_name, status, upload_date, c.company_name
                    FROM drawings d
                    LEFT JOIN customers c ON d.customer_id = c.id
                    WHERE d.product_category = ?
                    ORDER BY upload_date DESC
                """, (category,))

                results = cursor.fetchall()

                if not results:
                    print(f"类别 '{category}' 暂无图纸")
                    return

                print(f"共 {len(results)} 个图纸:\n")
                for drawing in results:
                    print(f"• {drawing[0]} ({drawing[1]})")
                    print(f"  客户: {drawing[3] or '未关联'}")
                    print(f"  上传时间: {drawing[2]}")
                    print()

            else:
                cursor.execute("""
                    SELECT product_category, COUNT(*) as count
                    FROM drawings
                    GROUP BY product_category
                    ORDER BY count DESC
                """)

                categories = cursor.fetchall()

                if not categories:
                    print("❌ 暂无图纸数据")
                    return

                print("图纸分类统计:\n")
                for category, count in categories:
                    print(f"• {category}: {count} 个图纸")

    def list_unlinked_customers(self):
        """列出未关联图纸的客户"""
        print("👥 未关联图纸的客户")
        print("=" * 50)

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT c.*, COUNT(d.id) as drawing_count
                FROM customers c
                LEFT JOIN drawings d ON c.id = d.customer_id
                GROUP BY c.id
                HAVING drawing_count = 0
                ORDER BY c.company_name
            """)

            customers = cursor.fetchall()

            if not customers:
                print("✅ 所有客户都有关联的图纸")
                return

            print(f"找到 {len(customers)} 个未关联图纸的客户:\n")

            for customer in customers:
                print(f"• {customer[1]} ({customer[3]})")
                print(f"  邮箱: {customer[2]}")
                print()

    def _handle_customer_command(self, command):
        """处理客户相关命令"""
        parts = command.split()
        if len(parts) == 1:
            self.list_all_customers()
        elif '--country' in parts:
            idx = parts.index('--country')
            if idx + 1 < len(parts):
                country = parts[idx + 1]
                self.list_all_customers(country_filter=country)
            else:
                print("❌ 请指定国家代码，如: customers --country CN")
        elif '--detail' in parts:
            idx = parts.index('--detail')
            if idx + 1 < len(parts):
                try:
                    customer_id = int(parts[idx + 1])
                    self.show_customer_detail(customer_id)
                except ValueError:
                    print("❌ 客户ID必须是数字")
            else:
                print("❌ 请指定客户ID，如: customers --detail 1")
        else:
            print("❌ 无效的customers命令格式")

    def _handle_drawing_command(self, command):
        """处理图纸相关命令"""
        parts = command.split()
        if len(parts) == 1:
            self.list_drawings_by_customer()
        elif '--customer' in parts:
            idx = parts.index('--customer')
            if idx + 1 < len(parts):
                try:
                    customer_id = int(parts[idx + 1])
                    self.list_drawings_by_customer(customer_id)
                except ValueError:
                    print("❌ 客户ID必须是数字")
            else:
                print("❌ 请指定客户ID")
        elif '--category' in parts:
            idx = parts.index('--category')
            if idx + 1 < len(parts):
                category = parts[idx + 1]
                self.list_drawings_by_category(category)
            else:
                print("❌ 请指定类别")
        else:
            print("❌ 无效的drawings命令格式")

    def _handle_search_command(self, command):
        """处理搜索命令"""
        parts = command.split()
        if len(parts) < 3:
            print("❌ 用法: search --customer KEYWORD 或 search --drawing KEYWORD")
            return

        search_type = parts[1]
        keyword = ' '.join(parts[2:])

        if search_type == '--customer':
            self.search_customers(keyword)
        elif search_type == '--drawing':
            self.search_drawings(keyword)
        else:
            print("❌ 搜索类型必须是 --customer 或 --drawing")

    def _handle_export_command(self, command):
        """处理导出命令"""
        parts = command.split()
        if len(parts) < 3:
            print("❌ 用法: export --customers FILE 或 export --drawings FILE")
            return

        export_type = parts[1]
        filename = parts[2]

        if export_type == '--customers':
            self._export_customers(filename)
        elif export_type == '--drawings':
            self._export_drawings(filename)
        else:
            print("❌ 导出类型必须是 --customers 或 --drawings")

    def _export_customers(self, filename):
        """导出客户列表"""
        customers = self.customer.get_all()

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("公司名称,联系人,邮箱,电话,国家,语言,首次联系,备注\n")
                for customer in customers:
                    f.write(f"{customer['company_name']},{customer.get('contact_name', '')},")
                    f.write(f"{customer.get('contact_email', '')},{customer.get('phone', '')},")
                    f.write(f"{customer.get('country', '')},{customer.get('language', '')},")
                    f.write(f"{customer.get('first_contact_date', '')},{customer.get('notes', '')}\n")

            print(f"✅ 客户列表已导出到: {filename}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")

    def _export_drawings(self, filename):
        """导出图纸列表"""
        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT d.drawing_name, d.product_category, d.status, d.upload_date,
                       c.company_name, d.file_path, d.notes
                FROM drawings d
                LEFT JOIN customers c ON d.customer_id = c.id
                ORDER BY d.upload_date DESC
            """)

            drawings = cursor.fetchall()

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("图纸名称,产品类别,状态,上传时间,客户,文件路径,备注\n")
                for drawing in drawings:
                    f.write(f"{drawing[0]},{drawing[1]},{drawing[2]},{drawing[3]},")
                    f.write(f"{drawing[4] or ''},{drawing[5]},{drawing[6] or ''}\n")

            print(f"✅ 图纸列表已导出到: {filename}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识库查询工具")
    parser.add_argument('--interactive', '-i', action='store_true', help='启动交互模式')
    parser.add_argument('--db', default='./data/db.sqlite', help='数据库文件路径')

    args = parser.parse_args()

    # 创建查询器
    query = KnowledgeBaseQuery(args.db)

    if args.interactive:
        query.run_interactive_mode()
    else:
        # 显示概览
        query.show_overview()

        # 显示一些有用的查询
        print("\n" + "="*50)
        print("💡 示例查询:")
        print("="*50)

        print("\n👥 未关联图纸的客户:")
        query.list_unlinked_customers()

        print("\n📄 图纸按类别统计:")
        query.list_drawings_by_category()

        print("\n💡 提示: 使用 --interactive 参数进入交互模式进行更多查询")

if __name__ == "__main__":
    main()