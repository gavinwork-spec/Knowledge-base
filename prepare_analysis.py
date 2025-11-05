#!/usr/bin/env python3
"""
分析准备脚本
为后续业务分析和提醒机制准备数据和分析框架
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from models import DatabaseManager, Customer, Drawing, FactoryQuote, Factory

class AnalysisPreparation:
    """分析准备器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)
        self.export_dir = Path("./data/analysis")
        self.export_dir.mkdir(exist_ok=True)

    def prepare_quote_trend_analysis(self):
        """准备报价趋势分析"""
        print("📈 准备报价趋势分析...")

        with self.db_manager:
            conn = self.db_manager.connect()

            # 导出工厂报价数据
            df_quotes = pd.read_sql_query("""
                SELECT
                    fq.id,
                    fq.factory_id,
                    f.factory_name,
                    fq.product_category,
                    fq.quote_date,
                    fq.price,
                    fq.moq,
                    fq.notes,
                    fq.created_at
                FROM factory_quotes fq
                LEFT JOIN factories f ON fq.factory_id = f.id
                ORDER BY fq.quote_date DESC, fq.product_category
            """, conn)

            if not df_quotes.empty:
                # 添加分析字段
                df_quotes['quote_date'] = pd.to_datetime(df_quotes['quote_date'], format='ISO8601', errors='coerce')
                df_quotes['price_per_unit'] = df_quotes['price']
                df_quotes['quarter'] = df_quotes['quote_date'].dt.to_period('Q')
                df_quotes['month'] = df_quotes['quote_date'].dt.to_period('M')

                # 导出分析数据
                quotes_file = self.export_dir / "factory_quotes_analysis.csv"
                df_quotes.to_csv(quotes_file, index=False, encoding='utf-8')
                print(f"  ✅ 报价数据已导出: {quotes_file}")

                # 生成趋势分析
                self._generate_quote_trends(df_quotes)

    def _generate_quote_trends(self, df_quotes):
        """生成报价趋势分析"""
        if df_quotes.empty:
            return

        print("  📊 生成报价趋势...")

        # 按产品类别的价格趋势
        category_trends = df_quotes.groupby('product_category').agg({
            'price': ['mean', 'min', 'max', 'count'],
            'quote_date': ['min', 'max']
        }).round(2)

        category_file = self.export_dir / "price_trends_by_category.csv"
        category_trends.to_csv(category_file, encoding='utf-8')
        print(f"  ✅ 分类价格趋势: {category_file}")

        # 按工厂的价格分析
        factory_trends = df_quotes.groupby('factory_name').agg({
            'price': ['mean', 'min', 'max', 'count'],
            'product_category': 'nunique'
        }).round(2)

        factory_file = self.export_dir / "price_trends_by_factory.csv"
        factory_trends.to_csv(factory_file, encoding='utf-8')
        print(f"  ✅ 工厂价格分析: {factory_file}")

    def prepare_customer_analysis(self):
        """准备客户分析"""
        print("👥 准备客户分析...")

        with self.db_manager:
            conn = self.db_manager.connect()

            # 客户活跃度分析
            df_customers = pd.read_sql_query("""
                SELECT
                    c.id,
                    c.company_name,
                    c.contact_email,
                    c.country,
                    c.language,
                    c.first_contact_date,
                    c.created_at,
                    COUNT(DISTINCT d.id) as drawing_count,
                    COUNT(DISTINCT d.product_category) as category_count,
                    MAX(d.upload_date) as last_drawing_date,
                    COUNT(DISTINCT ps.id) as process_count
                FROM customers c
                LEFT JOIN drawings d ON c.id = d.customer_id
                LEFT JOIN process_status ps ON c.id = ps.customer_id
                GROUP BY c.id
                ORDER BY c.company_name
            """, conn)

            if not df_customers.empty:
                # 添加分析字段
                df_customers['first_contact_date'] = pd.to_datetime(df_customers['first_contact_date'], format='ISO8601', errors='coerce')
                df_customers['created_at'] = pd.to_datetime(df_customers['created_at'], format='ISO8601', errors='coerce')
                df_customers['last_drawing_date'] = pd.to_datetime(df_customers['last_drawing_date'], format='ISO8601', errors='coerce')

                # 计算客户活跃度指标
                today = datetime.now()
                df_customers['days_since_first_contact'] = (today - df_customers['first_contact_date']).dt.days
                df_customers['days_since_last_activity'] = (today - df_customers['last_drawing_date']).dt.days
                df_customers['activity_level'] = df_customers['drawing_count'].apply(self._classify_activity)

                customers_file = self.export_dir / "customer_analysis.csv"
                df_customers.to_csv(customers_file, index=False, encoding='utf-8')
                print(f"  ✅ 客户分析数据: {customers_file}")

                # 生成客户细分
                self._generate_customer_segments(df_customers)

    def _classify_activity(self, drawing_count):
        """分类客户活跃度"""
        if drawing_count == 0:
            return "未活跃"
        elif drawing_count <= 5:
            return "低活跃"
        elif drawing_count <= 20:
            return "中活跃"
        else:
            return "高活跃"

    def _generate_customer_segments(self, df_customers):
        """生成客户细分"""
        if df_customers.empty:
            return

        print("  🎯 生成客户细分...")

        # 按国家和活跃度细分
        segments = df_customers.groupby(['country', 'activity_level']).agg({
            'company_name': 'count',
            'drawing_count': 'sum',
            'category_count': 'sum'
        }).rename(columns={'company_name': 'customer_count'})

        segments_file = self.export_dir / "customer_segments.csv"
        segments.to_csv(segments_file, encoding='utf-8')
        print(f"  ✅ 客户细分数据: {segments_file}")

    def prepare_drawing_analysis(self):
        """准备图纸分析"""
        print("📄 准备图纸分析...")

        with self.db_manager:
            conn = self.db_manager.connect()

            # 图纸活跃度分析
            df_drawings = pd.read_sql_query("""
                SELECT
                    d.id,
                    d.drawing_name,
                    d.product_category,
                    d.status,
                    d.upload_date,
                    d.created_at,
                    c.company_name,
                    c.country,
                    d.notes
                FROM drawings d
                LEFT JOIN customers c ON d.customer_id = c.id
                ORDER BY d.upload_date DESC
            """, conn)

            if not df_drawings.empty:
                # 时间分析
                df_drawings['upload_date'] = pd.to_datetime(df_drawings['upload_date'], format='ISO8601', errors='coerce')
                df_drawings['created_at'] = pd.to_datetime(df_drawings['created_at'], format='ISO8601', errors='coerce')
                df_drawings['month'] = df_drawings['upload_date'].dt.to_period('M')
                df_drawings['day_of_week'] = df_drawings['upload_date'].dt.day_name()

                drawings_file = self.export_dir / "drawing_analysis.csv"
                df_drawings.to_csv(drawings_file, index=False, encoding='utf-8')
                print(f"  ✅ 图纸分析数据: {drawings_file}")

                # 生成月度趋势
                self._generate_drawing_trends(df_drawings)

    def _generate_drawing_trends(self, df_drawings):
        """生成图纸趋势"""
        if df_drawings.empty:
            return

        print("  📈 生成图纸趋势...")

        # 月度上传趋势
        monthly_trends = df_drawings.groupby('month').agg({
            'id': 'count',
            'product_category': 'nunique',
            'company_name': 'nunique'
        }).rename(columns={'id': 'drawing_count'})

        monthly_file = self.export_dir / "drawing_monthly_trends.csv"
        monthly_trends.to_csv(monthly_file, encoding='utf-8')
        print(f"  ✅ 月度趋势数据: {monthly_file}")

        # 按类别的统计
        category_stats = df_drawings.groupby('product_category').agg({
            'id': 'count',
            'company_name': 'nunique',
            'status': lambda x: (x == 'pending').sum()
        }).rename(columns={'id': 'total_count', 'status': 'pending_count'})

        category_file = self.export_dir / "drawing_category_stats.csv"
        category_stats.to_csv(category_file, encoding='utf-8')
        print(f"  ✅ 类别统计数据: {category_file}")

    def prepare_factory_performance(self):
        """准备工厂绩效分析"""
        print("🏭 准备工厂绩效分析...")

        with self.db_manager:
            conn = self.db_manager.connect()

            # 工厂报价分析
            df_factory = pd.read_sql_query("""
                SELECT
                    f.id,
                    f.factory_name,
                    f.location,
                    f.capability,
                    f.cost_reference,
                    f.production_cycle,
                    COUNT(fq.id) as quote_count,
                    AVG(fq.price) as avg_price,
                    MIN(fq.price) as min_price,
                    MAX(fq.price) as max_price,
                    AVG(fq.moq) as avg_moq,
                    COUNT(DISTINCT fq.product_category) as category_count
                FROM factories f
                LEFT JOIN factory_quotes fq ON f.id = fq.factory_id
                GROUP BY f.id
                ORDER BY quote_count DESC, avg_price
            """, conn)

            if not df_factory.empty:
                factory_file = self.export_dir / "factory_performance.csv"
                df_factory.to_csv(factory_file, index=False, encoding='utf-8')
                print(f"  ✅ 工厂绩效数据: {factory_file}")

    def prepare_alert_metrics(self):
        """准备提醒指标"""
        print("🔔 准备提醒指标...")

        with self.db_manager:
            conn = self.db_manager.connect()

            alerts = []

            # 1. 长期未活跃客户
            df_inactive = pd.read_sql_query("""
                SELECT
                    c.id as customer_id,
                    c.company_name,
                    c.contact_email,
                    c.country,
                    MAX(d.upload_date) as last_activity,
                    COUNT(DISTINCT d.id) as total_drawings
                FROM customers c
                LEFT JOIN drawings d ON c.id = d.customer_id
                GROUP BY c.id
                HAVING last_activity < date('now', '-30 days') OR last_activity IS NULL
            """, conn)

            if not df_inactive.empty:
                for _, row in df_inactive.iterrows():
                    alerts.append({
                        'type': 'inactive_customer',
                        'customer_id': row['customer_id'],
                        'company_name': row['company_name'],
                        'email': row['contact_email'],
                        'severity': 'medium',
                        'message': f"客户 {row['company_name']} 长期未活跃",
                        'days_inactive': 30,
                        'data': row.to_dict()
                    })

            # 2. 未分类图纸过多
            unclassified_count = pd.read_sql_query("""
                SELECT COUNT(*) as count
                FROM drawings
                WHERE product_category = '未分类'
            """, conn).iloc[0]['count']

            if unclassified_count > 100:
                alerts.append({
                    'type': 'unclassified_drawings',
                    'severity': 'high',
                    'message': f"未分类图纸过多: {unclassified_count} 个",
                    'count': unclassified_count,
                    'threshold': 100
                })

            # 3. 报价波动检测 (当有足够数据时)
            quote_analysis = pd.read_sql_query("""
                SELECT
                    product_category,
                    AVG(price) as avg_price,
                    COUNT(*) as quote_count
                FROM factory_quotes
                WHERE quote_date >= date('now', '-90 days')
                GROUP BY product_category
                HAVING quote_count >= 3
            """, conn)

            if not quote_analysis.empty:
                # 与历史价格比较
                historical = pd.read_sql_query("""
                    SELECT
                        product_category,
                        AVG(price) as historical_avg
                    FROM factory_quotes
                    WHERE quote_date < date('now', '-90 days')
                    GROUP BY product_category
                """, conn)

                for _, recent in quote_analysis.iterrows():
                    historical_match = historical[historical['product_category'] == recent['product_category']]
                    if not historical_match.empty:
                        hist_avg = historical_match.iloc[0]['historical_avg']
                        price_change = (recent['avg_price'] - hist_avg) / hist_avg * 100

                        if abs(price_change) > 10:  # 10% 变化阈值
                            alerts.append({
                                'type': 'price_fluctuation',
                                'product_category': recent['product_category'],
                                'severity': 'high' if abs(price_change) > 20 else 'medium',
                                'message': f"{recent['product_category']} 价格波动 {price_change:.1f}%",
                                'recent_avg': recent['avg_price'],
                                'historical_avg': hist_avg,
                                'change_percent': price_change
                            })

            # 导出提醒数据
            if alerts:
                alerts_df = pd.DataFrame(alerts)
                alerts_file = self.export_dir / "current_alerts.csv"
                alerts_df.to_csv(alerts_file, index=False, encoding='utf-8')
                print(f"  ✅ 当前提醒: {alerts_file} ({len(alerts)} 个提醒)")
            else:
                print("  ✅ 当前无触发提醒")

    def generate_analysis_summary(self):
        """生成分析摘要"""
        print("📋 生成分析摘要...")

        summary_file = self.export_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("知识库分析准备摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 基础统计
            with self.db_manager:
                conn = self.db_manager.connect()
                cursor = conn.cursor()

                tables = ['customers', 'drawings', 'factories', 'factory_quotes']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    f.write(f"{table}: {count} 条记录\n")

            f.write(f"\n导出文件位置: {self.export_dir}\n")

            # 列出导出的文件
            analysis_files = list(self.export_dir.glob("*.csv"))
            if analysis_files:
                f.write(f"\n导出的分析文件:\n")
                for file_path in sorted(analysis_files):
                    f.write(f"- {file_path.name}\n")

        print(f"  ✅ 分析摘要: {summary_file}")

    def run_full_preparation(self):
        """运行完整的分析准备"""
        print("🚀 开始分析数据准备...")
        print("=" * 60)

        try:
            self.prepare_quote_trend_analysis()
            self.prepare_customer_analysis()
            self.prepare_drawing_analysis()
            self.prepare_factory_performance()
            self.prepare_alert_metrics()
            self.generate_analysis_summary()

            print("\n" + "=" * 60)
            print("✅ 分析数据准备完成!")
            print(f"📁 导出目录: {self.export_dir}")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 分析准备失败: {e}")
            raise

def main():
    """主函数"""
    analyzer = AnalysisPreparation()
    analyzer.run_full_preparation()

if __name__ == "__main__":
    main()