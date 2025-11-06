#!/usr/bin/env python3
"""
统计结果导出脚本
将分类/分析结果导出为 JSON/CSV 格式，便于后续可视化使用
"""

import sqlite3
import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class StatisticsExporter:
    """统计结果导出器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_path = db_path
        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'export_statistics.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('StatisticsExporter')

    def get_customers_by_status(self) -> Dict[str, Any]:
        """按客户状态统计"""
        self.logger.info("👥 统计客户状态分布...")

        try:
            conn = sqlite3.connect(self.db_path)

            # 基础状态统计
            query = """
            SELECT
                customer_status,
                COUNT(*) as count,
                COUNT(DISTINCT country) as countries,
                AVG(total_drawings) as avg_drawings,
                MAX(CASE WHEN last_inquiry_date IS NOT NULL THEN last_inquiry_date END) as last_inquiry
            FROM customers
            GROUP BY customer_status
            ORDER BY count DESC
            """

            df = pd.read_sql_query(query, conn)

            # 详细客户信息
            detailed_query = """
            SELECT
                company_name,
                contact_email,
                customer_status,
                customer_level,
                country,
                language,
                total_drawings,
                first_contact_date,
                last_inquiry_date,
                contact_frequency,
                created_at
            FROM customers
            ORDER BY customer_status, total_drawings DESC
            """

            detailed_df = pd.read_sql_query(detailed_query, conn)

            conn.close()

            result = {
                'summary': df.to_dict('records'),
                'detailed': detailed_df.to_dict('records'),
                'total_customers': len(detailed_df),
                'status_distribution': df.set_index('customer_status')['count'].to_dict()
            }

            self.logger.info(f"✅ 客户状态统计完成: {len(df)} 个状态")
            return result

        except Exception as e:
            self.logger.error(f"❌ 客户状态统计失败: {e}")
            return {'error': str(e)}

    def get_drawings_by_category(self) -> Dict[str, Any]:
        """按产品类别统计图纸"""
        self.logger.info("📊 统计图纸分类分布...")

        try:
            conn = sqlite3.connect(self.db_path)

            # 基础分类统计
            query = """
            SELECT
                product_category,
                COUNT(*) as count,
                COUNT(DISTINCT customer_id) as unique_customers,
                AVG(classification_confidence) as avg_confidence,
                COUNT(CASE WHEN standard_or_custom = 1 THEN 1 END) as custom_count,
                COUNT(CASE WHEN standard_or_custom = 0 THEN 1 END) as standard_count,
                MIN(classification_date) as first_classified,
                MAX(classification_date) as last_classified
            FROM drawings
            GROUP BY product_category
            ORDER BY count DESC
            """

            df = pd.read_sql_query(query, conn)

            # 标准件vs定制件统计
            standard_custom_query = """
            SELECT
                standard_or_custom,
                COUNT(*) as count,
                COUNT(DISTINCT product_category) as categories
            FROM drawings
            WHERE standard_or_custom IS NOT NULL
            GROUP BY standard_or_custom
            """

            standard_custom_df = pd.read_sql_query(standard_custom_query, conn)

            # 按时间统计分类趋势
            time_trend_query = """
            SELECT
                DATE(classification_date) as classification_day,
                product_category,
                COUNT(*) as daily_count
            FROM drawings
            WHERE classification_date IS NOT NULL
            GROUP BY DATE(classification_date), product_category
            ORDER BY classification_day DESC
            LIMIT 100
            """

            time_trend_df = pd.read_sql_query(time_trend_query, conn)

            # 按数据源统计
            source_query = """
            SELECT
                data_source,
                COUNT(*) as count,
                COUNT(DISTINCT product_category) as categories
            FROM drawings
            WHERE data_source IS NOT NULL AND data_source != 'unknown'
            GROUP BY data_source
            ORDER BY count DESC
            """

            source_df = pd.read_sql_query(source_query, conn)

            conn.close()

            result = {
                'category_distribution': df.to_dict('records'),
                'standard_vs_custom': standard_custom_df.to_dict('records'),
                'time_trend': time_trend_df.to_dict('records'),
                'data_sources': source_df.to_dict('records'),
                'total_drawings': df['count'].sum(),
                'classified_drawings': df[df['product_category'] != '未分类']['count'].sum() if len(df) > 0 else 0,
                'classification_rate': (df[df['product_category'] != '未分类']['count'].sum() / df['count'].sum() * 100) if len(df) > 0 and df['count'].sum() > 0 else 0
            }

            self.logger.info(f"✅ 图纸分类统计完成: {len(df)} 个类别")
            return result

        except Exception as e:
            self.logger.error(f"❌ 图纸分类统计失败: {e}")
            return {'error': str(e)}

    def get_factory_performance_stats(self) -> Dict[str, Any]:
        """工厂表现统计"""
        self.logger.info("🏭 统计工厂表现...")

        try:
            conn = sqlite3.connect(self.db_path)

            # 工厂报价统计
            factory_query = """
            SELECT
                f.id as factory_id,
                f.factory_name,
                f.location,
                f.capability,
                COUNT(fq.id) as total_quotes,
                COUNT(DISTINCT fq.product_category) as unique_categories,
                AVG(fq.price) as avg_price,
                MIN(fq.price) as min_price,
                MAX(fq.price) as max_price,
                AVG(fq.moq) as avg_moq,
                MIN(fq.quote_date) as first_quote,
                MAX(fq.quote_date) as last_quote,
                COUNT(DISTINCT DATE(fq.quote_date, 'start of month')) as active_months
            FROM factories f
            LEFT JOIN factory_quotes fq ON f.id = fq.factory_id
            GROUP BY f.id, f.factory_name, f.location, f.capability
            ORDER BY total_quotes DESC
            """

            factory_df = pd.read_sql_query(factory_query, conn)

            # 按产品类别统计工厂报价
            category_query = """
            SELECT
                f.factory_name,
                fq.product_category,
                COUNT(fq.id) as quote_count,
                AVG(fq.price) as avg_price,
                MIN(fq.price) as min_price,
                MAX(fq.price) as max_price
            FROM factories f
            INNER JOIN factory_quotes fq ON f.id = fq.factory_id
            GROUP BY f.id, f.factory_name, fq.product_category
            ORDER BY f.factory_name, quote_count DESC
            """

            category_df = pd.read_sql_query(category_query, conn)

            conn.close()

            result = {
                'factory_summary': factory_df.to_dict('records'),
                'factory_by_category': category_df.to_dict('records'),
                'total_factories': len(factory_df),
                'active_factories': len(factory_df[factory_df['total_quotes'] > 0])
            }

            self.logger.info(f"✅ 工厂表现统计完成: {len(factory_df)} 个工厂")
            return result

        except Exception as e:
            self.logger.error(f"❌ 工厂表现统计失败: {e}")
            return {'error': str(e)}

    def get_temporal_analysis(self) -> Dict[str, Any]:
        """时间维度分析"""
        self.logger.info("📅 统计时间维度数据...")

        try:
            conn = sqlite3.connect(self.db_path)

            # 客户注册趋势
            customer_trend_query = """
            SELECT
                DATE(created_at) as registration_date,
                COUNT(*) as new_customers,
                COUNT(DISTINCT country) as new_countries
            FROM customers
            WHERE created_at IS NOT NULL
            GROUP BY DATE(created_at)
            ORDER BY registration_date DESC
            LIMIT 90
            """

            customer_trend_df = pd.read_sql_query(customer_trend_query, conn)

            # 图纸上传趋势
            drawing_upload_query = """
            SELECT
                DATE(upload_date) as upload_date,
                COUNT(*) as uploaded_drawings,
                COUNT(DISTINCT customer_id) as unique_customers,
                COUNT(DISTINCT product_category) as unique_categories
            FROM drawings
            WHERE upload_date IS NOT NULL
            GROUP BY DATE(upload_date)
            ORDER BY upload_date DESC
            LIMIT 90
            """

            drawing_upload_df = pd.read_sql_query(drawing_upload_query, conn)

            # 报价时间趋势
            quote_trend_query = """
            SELECT
                DATE(quote_date) as quote_date,
                COUNT(*) as quotes_count,
                COUNT(DISTINCT factory_id) as unique_factories,
                AVG(price) as avg_price
            FROM factory_quotes
            WHERE quote_date IS NOT NULL
            GROUP BY DATE(quote_date)
            ORDER BY quote_date DESC
            LIMIT 90
            """

            quote_trend_df = pd.read_sql_query(quote_trend_query, conn)

            conn.close()

            result = {
                'customer_registration_trend': customer_trend_df.to_dict('records'),
                'drawing_upload_trend': drawing_upload_df.to_dict('records'),
                'quote_trend': quote_trend_df.to_dict('records')
            }

            self.logger.info("✅ 时间维度分析完成")
            return result

        except Exception as e:
            self.logger.error(f"❌ 时间维度分析失败: {e}")
            return {'error': str(e)}

    def get_quality_metrics(self) -> Dict[str, Any]:
        """数据质量指标"""
        self.logger.info("🔍 计算数据质量指标...")

        try:
            conn = sqlite3.connect(self.db_path)

            # 客户数据质量
            customer_quality_query = """
            SELECT
                COUNT(*) as total_customers,
                COUNT(CASE WHEN contact_email IS NOT NULL AND contact_email != '' THEN 1 END) as has_email,
                COUNT(CASE WHEN company_name IS NOT NULL AND company_name != '' THEN 1 END) as has_company_name,
                COUNT(CASE WHEN country IS NOT NULL AND country != '' THEN 1 END) as has_country,
                COUNT(CASE WHEN last_inquiry_date IS NOT NULL THEN 1 END) as has_last_inquiry,
                COUNT(CASE WHEN total_drawings > 0 THEN 1 END) as has_drawings
            FROM customers
            """

            customer_quality = pd.read_sql_query(customer_quality_query, conn).iloc[0].to_dict()

            # 图纸数据质量
            drawing_quality_query = """
            SELECT
                COUNT(*) as total_drawings,
                COUNT(CASE WHEN drawing_name IS NOT NULL AND drawing_name != '' THEN 1 END) as has_name,
                COUNT(CASE WHEN file_path IS NOT NULL AND file_path != '' THEN 1 END) as has_file_path,
                COUNT(CASE WHEN customer_id IS NOT NULL THEN 1 END) as has_customer,
                COUNT(CASE WHEN product_category IS NOT NULL AND product_category != '' THEN 1 END) as has_category,
                COUNT(CASE WHEN is_classified = 1 THEN 1 END) as is_classified,
                AVG(classification_confidence) as avg_confidence
            FROM drawings
            """

            drawing_quality = pd.read_sql_query(drawing_quality_query, conn).iloc[0].to_dict()

            # 工厂数据质量
            factory_quality_query = """
            SELECT
                COUNT(*) as total_factories,
                COUNT(CASE WHEN factory_name IS NOT NULL AND factory_name != '' THEN 1 END) as has_name,
                COUNT(CASE WHEN location IS NOT NULL AND location != '' THEN 1 END) as has_location,
                COUNT(CASE WHEN capability IS NOT NULL AND capability != '' THEN 1 END) as has_capability
            FROM factories
            """

            factory_quality = pd.read_sql_query(factory_quality_query, conn).iloc[0].to_dict()

            conn.close()

            # 计算质量分数
            customer_quality_score = (
                (customer_quality['has_email'] / customer_quality['total_customers'] * 40) +
                (customer_quality['has_company_name'] / customer_quality['total_customers'] * 30) +
                (customer_quality['has_country'] / customer_quality['total_customers'] * 15) +
                (customer_quality['has_last_inquiry'] / customer_quality['total_customers'] * 15)
            )

            drawing_quality_score = (
                (drawing_quality['has_name'] / drawing_quality['total_drawings'] * 25) +
                (drawing_quality['has_file_path'] / drawing_quality['total_drawings'] * 25) +
                (drawing_quality['has_customer'] / drawing_quality['total_drawings'] * 25) +
                (drawing_quality['has_category'] / drawing_quality['total_drawings'] * 25)
            )

            factory_quality_score = (
                (factory_quality['has_name'] / factory_quality['total_factories'] * 40) +
                (factory_quality['has_location'] / factory_quality['total_factories'] * 30) +
                (factory_quality['has_capability'] / factory_quality['total_factories'] * 30)
            )

            result = {
                'customer_quality': {
                    **customer_quality,
                    'quality_score': round(customer_quality_score, 2),
                    'email_completeness': round(customer_quality['has_email'] / customer_quality['total_customers'] * 100, 2),
                    'name_completeness': round(customer_quality['has_company_name'] / customer_quality['total_customers'] * 100, 2)
                },
                'drawing_quality': {
                    **drawing_quality,
                    'quality_score': round(drawing_quality_score, 2),
                    'classification_rate': round(drawing_quality['is_classified'] / drawing_quality['total_drawings'] * 100, 2),
                    'category_completeness': round(drawing_quality['has_category'] / drawing_quality['total_drawings'] * 100, 2)
                },
                'factory_quality': {
                    **factory_quality,
                    'quality_score': round(factory_quality_score, 2),
                    'name_completeness': round(factory_quality['has_name'] / factory_quality['total_factories'] * 100, 2)
                },
                'overall_quality': round((customer_quality_score + drawing_quality_score + factory_quality_score) / 3, 2)
            }

            self.logger.info("✅ 数据质量指标计算完成")
            return result

        except Exception as e:
            self.logger.error(f"❌ 数据质量指标计算失败: {e}")
            return {'error': str(e)}

    def convert_for_json(self, obj):
        """转换对象为JSON可序列化格式"""
        import numpy as np

        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, 'isna') and hasattr(obj, 'any'):
            # 处理pandas布尔数组和NA值
            try:
                if obj.any() if hasattr(obj, 'any') else False:
                    return True
                elif obj.all() if hasattr(obj, 'all') else False:
                    return True
                else:
                    return None
            except:
                return None
        elif pd.isna(obj) if not isinstance(obj, (list, dict)) else False:
            return None
        elif isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        else:
            return obj

    def export_to_files(self, statistics: Dict[str, Any]) -> Dict[str, str]:
        """导出统计结果到文件"""
        self.logger.info("💾 导出统计结果...")

        output_dir = Path("./data/processed")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}

        try:
            # 1. 导出客户状态统计
            if 'customers_by_status' in statistics:
                customers_file = output_dir / f"customers_by_status_{timestamp}.json"
                with open(customers_file, 'w', encoding='utf-8') as f:
                    json.dump(self.convert_for_json(statistics['customers_by_status']), f, ensure_ascii=False, indent=2)
                exported_files['customers_by_status'] = str(customers_file)

                # 同时导出CSV格式
                customers_df = pd.DataFrame(statistics['customers_by_status']['detailed'])
                customers_csv_file = output_dir / f"customers_detailed_{timestamp}.csv"
                customers_df.to_csv(customers_csv_file, index=False, encoding='utf-8-sig')
                exported_files['customers_detailed_csv'] = str(customers_csv_file)

            # 2. 导出图纸分类统计
            if 'drawings_by_category' in statistics:
                drawings_file = output_dir / f"drawings_by_category_{timestamp}.json"
                with open(drawings_file, 'w', encoding='utf-8') as f:
                    json.dump(self.convert_for_json(statistics['drawings_by_category']), f, ensure_ascii=False, indent=2)
                exported_files['drawings_by_category'] = str(drawings_file)

                # 导出分类分布CSV
                category_df = pd.DataFrame(statistics['drawings_by_category']['category_distribution'])
                category_csv_file = output_dir / f"drawings_category_distribution_{timestamp}.csv"
                category_df.to_csv(category_csv_file, index=False, encoding='utf-8-sig')
                exported_files['drawings_category_csv'] = str(category_csv_file)

            # 3. 导出工厂表现统计
            if 'factory_performance' in statistics:
                factory_file = output_dir / f"factory_performance_{timestamp}.json"
                with open(factory_file, 'w', encoding='utf-8') as f:
                    json.dump(self.convert_for_json(statistics['factory_performance']), f, ensure_ascii=False, indent=2)
                exported_files['factory_performance'] = str(factory_file)

                # 导出工厂摘要CSV
                factory_df = pd.DataFrame(statistics['factory_performance']['factory_summary'])
                factory_csv_file = output_dir / f"factory_summary_{timestamp}.csv"
                factory_df.to_csv(factory_csv_file, index=False, encoding='utf-8-sig')
                exported_files['factory_summary_csv'] = str(factory_csv_file)

            # 4. 导出时间维度分析
            if 'temporal_analysis' in statistics:
                temporal_file = output_dir / f"temporal_analysis_{timestamp}.json"
                with open(temporal_file, 'w', encoding='utf-8') as f:
                    json.dump(self.convert_for_json(statistics['temporal_analysis']), f, ensure_ascii=False, indent=2)
                exported_files['temporal_analysis'] = str(temporal_file)

            # 5. 导出数据质量指标
            if 'quality_metrics' in statistics:
                quality_file = output_dir / f"quality_metrics_{timestamp}.json"
                with open(quality_file, 'w', encoding='utf-8') as f:
                    json.dump(self.convert_for_json(statistics['quality_metrics']), f, ensure_ascii=False, indent=2)
                exported_files['quality_metrics'] = str(quality_file)

            # 6. 导出综合统计报告
            comprehensive_report = {
                'generated_at': datetime.now().isoformat(),
                'statistics_summary': {
                    'customers_by_status': statistics.get('customers_by_status', {}).get('total_customers', 0),
                    'drawings_by_category': statistics.get('drawings_by_category', {}).get('total_drawings', 0),
                    'factory_performance': statistics.get('factory_performance', {}).get('total_factories', 0),
                    'overall_quality_score': statistics.get('quality_metrics', {}).get('overall_quality', 0)
                },
                'detailed_statistics': statistics
            }

            comprehensive_file = output_dir / f"comprehensive_statistics_{timestamp}.json"
            with open(comprehensive_file, 'w', encoding='utf-8') as f:
                json.dump(self.convert_for_json(comprehensive_report), f, ensure_ascii=False, indent=2)
            exported_files['comprehensive_statistics'] = str(comprehensive_file)

            self.logger.info(f"✅ 统计结果导出完成: {len(exported_files)} 个文件")
            return exported_files

        except Exception as e:
            self.logger.error(f"❌ 导出失败: {e}")
            return {}

    def run_export(self) -> Dict[str, Any]:
        """运行完整统计导出"""
        self.logger.info("🚀 开始统计结果导出...")

        start_time = datetime.now()

        try:
            # 收集所有统计数据
            statistics = {
                'customers_by_status': self.get_customers_by_status(),
                'drawings_by_category': self.get_drawings_by_category(),
                'factory_performance': self.get_factory_performance_stats(),
                'temporal_analysis': self.get_temporal_analysis(),
                'quality_metrics': self.get_quality_metrics()
            }

            # 导出文件
            exported_files = self.export_to_files(statistics)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # 检查是否有错误
            errors = {key: value.get('error') for key, value in statistics.items() if isinstance(value, dict) and 'error' in value}

            result = {
                'success': len(errors) == 0,
                'processing_time': processing_time,
                'statistics_summary': {
                    'customers_count': statistics.get('customers_by_status', {}).get('total_customers', 0),
                    'drawings_count': statistics.get('drawings_by_category', {}).get('total_drawings', 0),
                    'factories_count': statistics.get('factory_performance', {}).get('total_factories', 0),
                    'overall_quality': statistics.get('quality_metrics', {}).get('overall_quality', 0)
                },
                'exported_files': exported_files,
                'errors': errors if errors else None
            }

            if result['success']:
                self.logger.info(f"✅ 统计导出完成! 耗时: {processing_time:.2f}秒")
            else:
                self.logger.warning(f"⚠️ 部分统计导出失败，错误: {errors}")

            return result

        except Exception as e:
            self.logger.error(f"❌ 统计导出失败: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='统计结果导出工具')
    parser.add_argument('--db-path', default='./data/db.sqlite', help='数据库文件路径')
    parser.add_argument('--output-dir', default='./data/processed', help='输出目录')
    parser.add_argument('--format', choices=['json', 'csv', 'all'], default='all', help='导出格式')
    parser.add_argument('--stats-type', choices=['customers', 'drawings', 'factory', 'temporal', 'quality', 'all'], default='all', help='统计类型')

    args = parser.parse_args()

    exporter = StatisticsExporter(args.db_path)

    result = exporter.run_export()

    if result['success']:
        print("✅ 统计结果导出完成!")
        print(f"📊 客户数: {result['statistics_summary']['customers_count']}")
        print(f"📄 图纸数: {result['statistics_summary']['drawings_count']}")
        print(f"🏭 工厂数: {result['statistics_summary']['factories_count']}")
        print(f"📈 数据质量: {result['statistics_summary']['overall_quality']}/100")
        print(f"⏱️ 处理时间: {result['processing_time']:.2f}秒")

        print("\n📄 导出文件:")
        for file_type, file_path in result['exported_files'].items():
            print(f"  {file_type}: {file_path}")
    else:
        print(f"❌ 统计导出失败: {result.get('error', '未知错误')}")
        if result.get('errors'):
            print("详细错误:")
            for stat_type, error in result['errors'].items():
                print(f"  {stat_type}: {error}")

if __name__ == "__main__":
    main()