#!/usr/bin/env python3
"""
数据验证脚本
详细检查数据库完整性、数据质量和关联关系
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from models import DatabaseManager, Customer, Drawing

class DataValidator:
    """数据验证器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)
        self.issues = []
        self.warnings = []
        self.stats = {}

    def run_full_validation(self):
        """运行完整的数据验证"""
        print("=" * 80)
        print("🔍 知识库数据完整性验证报告")
        print("=" * 80)
        print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 1. 基础统计
        self._collect_basic_stats()
        self._print_basic_stats()

        # 2. 客户数据质量检查
        print("\n👥 客户数据质量检查:")
        self._validate_customers()

        # 3. 图纸数据质量检查
        print("\n📄 图纸数据质量检查:")
        self._validate_drawings()

        # 4. 关联关系检查
        print("\n🔗 关联关系检查:")
        self._validate_relationships()

        # 5. 数据一致性问题
        print("\n⚠️ 数据一致性问题:")
        self._check_data_consistency()

        # 6. 生成建议
        print("\n💡 改进建议:")
        self._generate_recommendations()

        # 7. 导出报告
        self._export_report()

        print("\n" + "=" * 80)
        print("✅ 数据验证完成")
        print(f"发现 {len(self.issues)} 个问题，{len(self.warnings)} 个警告")
        print("=" * 80)

    def _collect_basic_stats(self):
        """收集基础统计数据"""
        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 各表记录数
            tables = ['customers', 'factories', 'drawings', 'factory_quotes', 'specifications', 'process_status']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                self.stats[f'{table}_count'] = cursor.fetchone()[0]

            # 客户统计
            cursor.execute("SELECT COUNT(DISTINCT company_name) FROM customers")
            self.stats['unique_companies'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT contact_email) FROM customers WHERE contact_email IS NOT NULL")
            self.stats['unique_emails'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM customers WHERE contact_email IS NULL OR contact_email = ''")
            self.stats['customers_without_email'] = cursor.fetchone()[0]

            # 图纸统计
            cursor.execute("SELECT COUNT(DISTINCT product_category) FROM drawings")
            self.stats['unique_categories'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM drawings WHERE customer_id IS NULL")
            self.stats['drawings_without_customer'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM drawings WHERE product_category = '未分类'")
            self.stats['uncategorized_drawings'] = cursor.fetchone()[0]

    def _print_basic_stats(self):
        """打印基础统计信息"""
        print(f"📊 基础统计:")
        print(f"  客户记录: {self.stats.get('customers_count', 0)} 条")
        print(f"  图纸记录: {self.stats.get('drawings_count', 0)} 条")
        print(f"  工厂记录: {self.stats.get('factories_count', 0)} 条")
        print(f"  报价记录: {self.stats.get('factory_quotes_count', 0)} 条")
        print(f"  规格记录: {self.stats.get('specifications_count', 0)} 条")
        print(f"  流程记录: {self.stats.get('process_status_count', 0)} 条")
        print(f"  独立公司: {self.stats.get('unique_companies', 0)} 个")
        print(f"  独立邮箱: {self.stats.get('unique_emails', 0)} 个")

    def _validate_customers(self):
        """验证客户数据"""
        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 检查邮箱格式
            cursor.execute("""
                SELECT id, company_name, contact_email
                FROM customers
                WHERE contact_email IS NOT NULL
                AND contact_email != ''
            """)

            customers = cursor.fetchall()
            invalid_emails = []
            duplicate_emails = {}

            for cust_id, company, email in customers:
                # 检查邮箱格式
                if not self._is_valid_email(email):
                    invalid_emails.append((cust_id, company, email))

                # 检查重复邮箱
                if email in duplicate_emails:
                    duplicate_emails[email].append((cust_id, company))
                else:
                    duplicate_emails[email] = [(cust_id, company)]

            # 报告问题
            if invalid_emails:
                self.issues.append(f"发现 {len(invalid_emails)} 个无效邮箱格式")
                print(f"  ❌ 无效邮箱格式: {len(invalid_emails)} 个")
                for cust_id, company, email in invalid_emails[:3]:
                    print(f"     - {company}: {email}")

            duplicate_count = sum(1 for email, customers in duplicate_emails.items() if len(customers) > 1)
            if duplicate_count > 0:
                self.warnings.append(f"发现 {duplicate_count} 个重复邮箱")
                print(f"  ⚠️  重复邮箱: {duplicate_count} 个")

            # 检查缺失关键字段
            cursor.execute("SELECT COUNT(*) FROM customers WHERE company_name IS NULL OR company_name = ''")
            missing_company = cursor.fetchone()[0]
            if missing_company > 0:
                self.issues.append(f"发现 {missing_company} 个客户缺少公司名称")
                print(f"  ❌ 缺少公司名称: {missing_company} 个")

            cursor.execute("SELECT COUNT(*) FROM customers WHERE contact_email IS NULL OR contact_email = ''")
            missing_email = cursor.fetchone()[0]
            if missing_email > 0:
                self.warnings.append(f"发现 {missing_email} 个客户缺少邮箱")
                print(f"  ⚠️  缺少邮箱: {missing_email} 个")

            if not invalid_emails and missing_company == 0:
                print("  ✅ 客户数据质量良好")

    def _validate_drawings(self):
        """验证图纸数据"""
        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 检查文件路径
            cursor.execute("SELECT COUNT(*) FROM drawings WHERE file_path IS NULL OR file_path = ''")
            missing_path = cursor.fetchone()[0]
            if missing_path > 0:
                self.issues.append(f"发现 {missing_path} 个图纸缺少文件路径")
                print(f"  ❌ 缺少文件路径: {missing_path} 个")

            # 检查文件是否存在
            cursor.execute("SELECT id, drawing_name, file_path FROM drawings")
            drawings = cursor.fetchall()
            missing_files = []

            for draw_id, name, path in drawings:
                if path and not Path(path).exists():
                    missing_files.append((draw_id, name, path))

            if missing_files:
                self.warnings.append(f"发现 {len(missing_files)} 个图纸文件不存在")
                print(f"  ⚠️  文件不存在: {len(missing_files)} 个")
                for draw_id, name, path in missing_files[:3]:
                    print(f"     - {name}: {path}")

            # 检查产品分类
            uncategorized = self.stats.get('uncategorized_drawings', 0)
            if uncategorized > 0:
                self.warnings.append(f"发现 {uncategorized} 个未分类图纸")
                print(f"  ⚠️  未分类图纸: {uncategorized} 个")

            # 检查重复图纸
            cursor.execute("""
                SELECT drawing_name, COUNT(*) as count
                FROM drawings
                GROUP BY drawing_name
                HAVING count > 1
            """)
            duplicates = cursor.fetchall()
            if duplicates:
                self.warnings.append(f"发现 {len(duplicates)} 组重复图纸名称")
                print(f"  ⚠️  重复图纸名称: {len(duplicates)} 组")

            if missing_path == 0 and len(missing_files) == 0:
                print("  ✅ 图纸数据基本完整")

    def _validate_relationships(self):
        """验证关联关系"""
        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 检查孤立图纸（没有关联客户的图纸）
            cursor.execute("""
                SELECT COUNT(*) FROM drawings d
                LEFT JOIN customers c ON d.customer_id = c.id
                WHERE d.customer_id IS NOT NULL AND c.id IS NULL
            """)
            orphan_drawings = cursor.fetchone()[0]
            if orphan_drawings > 0:
                self.issues.append(f"发现 {orphan_drawings} 个孤立图纸（引用不存在的客户）")
                print(f"  ❌ 孤立图纸: {orphan_drawings} 个")

            # 检查客户关联率
            total_drawings = self.stats.get('drawings_count', 0)
            unlinked_drawings = self.stats.get('drawings_without_customer', 0)
            if total_drawings > 0:
                linkage_rate = (total_drawings - unlinked_drawings) / total_drawings * 100
                print(f"  📊 客户关联率: {linkage_rate:.1f}% ({total_drawings - unlinked_drawings}/{total_drawings})")

                if linkage_rate < 50:
                    self.warnings.append(f"客户关联率较低 ({linkage_rate:.1f}%)，建议增强自动匹配功能")

            # 检查流程状态关联
            cursor.execute("""
                SELECT COUNT(*) FROM process_status ps
                LEFT JOIN drawings d ON ps.drawing_id = d.id
                WHERE ps.drawing_id IS NOT NULL AND d.id IS NULL
            """)
            orphan_status = cursor.fetchone()[0]
            if orphan_status > 0:
                self.issues.append(f"发现 {orphan_status} 个孤立流程状态")
                print(f"  ❌ 孤立流程状态: {orphan_status} 个")

    def _check_data_consistency(self):
        """检查数据一致性"""
        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 检查日期格式
            cursor.execute("""
                SELECT COUNT(*) FROM customers
                WHERE first_contact_date IS NOT NULL
                AND first_contact_date != ''
                AND date(first_contact_date) IS NULL
            """)
            invalid_dates = cursor.fetchone()[0]
            if invalid_dates > 0:
                self.warnings.append(f"发现 {invalid_dates} 个无效日期格式")

            # 检查状态值
            valid_statuses = ['pending', 'confirmed', 'approved', 'rejected', 'archived']
            cursor.execute(f"""
                SELECT COUNT(*) FROM drawings
                WHERE status NOT IN ({','.join(['?']*len(valid_statuses))})
            """, valid_statuses)
            invalid_statuses = cursor.fetchone()[0]
            if invalid_statuses > 0:
                self.issues.append(f"发现 {invalid_statuses} 个无效状态值")

            if not self.issues:
                print("  ✅ 数据一致性良好")

    def _generate_recommendations(self):
        """生成改进建议"""
        recommendations = []

        # 基于发现的问题生成建议
        if self.stats.get('customers_without_email', 0) > 0:
            recommendations.append("建议补充客户邮箱信息，提高客户匹配准确性")

        if self.stats.get('uncategorized_drawings', 0) > 100:
            recommendations.append("建议对图纸进行分类，提高知识库组织性")

        unlinked_rate = self.stats.get('drawings_without_customer', 0) / max(1, self.stats.get('drawings_count', 1))
        if unlinked_rate > 0.8:
            recommendations.append("建议增强从文件名推断客户信息的功能，提高关联率")
            recommendations.append("考虑手动关联重要客户的图纸文件")

        if len(recommendations) == 0:
            print("  ✅ 数据质量良好，无需特别改进")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

    def _export_report(self):
        """导出详细报告"""
        report_dir = Path("./data/reports")
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"validation_report_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("知识库数据验证报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("基础统计:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")

            f.write(f"\n问题汇总:\n")
            f.write(f"  问题数量: {len(self.issues)}\n")
            f.write(f"  警告数量: {len(self.warnings)}\n")

            if self.issues:
                f.write(f"\n发现的问题:\n")
                for issue in self.issues:
                    f.write(f"  - {issue}\n")

            if self.warnings:
                f.write(f"\n警告信息:\n")
                for warning in self.warnings:
                    f.write(f"  - {warning}\n")

        print(f"  📄 详细报告已导出: {report_file}")

    @staticmethod
    def _is_valid_email(email):
        """检查邮箱格式是否有效"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

def main():
    """主函数"""
    validator = DataValidator()
    validator.run_full_validation()

if __name__ == "__main__":
    main()