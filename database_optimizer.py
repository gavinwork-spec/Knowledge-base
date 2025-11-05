#!/usr/bin/env python3
"""
数据库性能分析和索引优化脚本
分析查询性能，提供索引建议
"""

import sqlite3
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging

class DatabaseOptimizer:
    """数据库优化器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DatabaseOptimizer')

    def analyze_table_sizes(self) -> Dict[str, int]:
        """分析表大小"""
        self.logger.info("📊 分析表大小...")

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, sql FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)

        tables = cursor.fetchall()
        table_sizes = {}

        for table_name, create_sql in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table_sizes[table_name] = count

        return table_sizes

    def get_existing_indexes(self) -> List[Dict[str, Any]]:
        """获取现有索引信息"""
        self.logger.info("📋 获取现有索引信息...")

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, tbl_name, sql FROM sqlite_master
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)

        indexes = []
        for index_name, table_name, sql in cursor.fetchall():
            indexes.append({
                'name': index_name,
                'table': table_name,
                'sql': sql,
                'is_unique': 'UNIQUE' in (sql or '').upper()
            })

        return indexes

    def analyze_query_performance(self) -> Dict[str, Any]:
        """分析常用查询性能"""
        self.logger.info("⚡ 分析查询性能...")

        cursor = self.conn.cursor()

        # 定义常用查询
        common_queries = [
            {
                'name': '客户按邮箱查询',
                'query': 'EXPLAIN QUERY PLAN SELECT * FROM customers WHERE contact_email = ?',
                'params': ['test@example.com']
            },
            {
                'name': '客户按公司名查询',
                'query': 'EXPLAIN QUERY PLAN SELECT * FROM customers WHERE company_name = ?',
                'params': ['Test Company']
            },
            {
                'name': '客户邮箱+公司名查询',
                'query': 'EXPLAIN QUERY PLAN SELECT * FROM customers WHERE company_name = ? AND contact_email = ?',
                'params': ['Test Company', 'test@example.com']
            },
            {
                'name': '图纸按客户查询',
                'query': 'EXPLAIN QUERY PLAN SELECT * FROM drawings WHERE customer_id = ?',
                'params': [1]
            },
            {
                'name': '图纸按分类查询',
                'query': 'EXPLAIN QUERY PLAN SELECT * FROM drawings WHERE product_category = ?',
                'params': ['螺栓']
            },
            {
                'name': '图纸客户+分类查询',
                'query': 'EXPLAIN QUERY PLAN SELECT * FROM drawings WHERE customer_id = ? AND product_category = ?',
                'params': [1, '螺栓']
            },
            {
                'name': '报价按工厂查询',
                'query': 'EXPLAIN QUERY PLAN SELECT * FROM factory_quotes WHERE factory_id = ?',
                'params': [1]
            },
            {
                'name': '规格按分类查询',
                'query': 'EXPLAIN QUERY PLAN SELECT * FROM specifications WHERE product_category = ?',
                'params': ['螺栓']
            }
        ]

        query_analysis = {}

        for query_info in common_queries:
            try:
                # 执行查询计划分析
                cursor.execute(query_info['query'], query_info['params'])
                explain_results = cursor.fetchall()

                # 分析查询计划
                uses_index = any('USING INDEX' in str(row) for row in explain_results)
                scan_type = 'INDEX SCAN' if uses_index else 'TABLE SCAN'

                query_analysis[query_info['name']] = {
                    'query': query_info['query'].replace('EXPLAIN QUERY PLAN ', ''),
                    'uses_index': uses_index,
                    'scan_type': scan_type,
                    'explain_plan': [str(row) for row in explain_results]
                }

            except Exception as e:
                self.logger.warning(f"查询分析失败 {query_info['name']}: {e}")
                query_analysis[query_info['name']] = {
                    'error': str(e)
                }

        return query_analysis

    def test_query_performance(self) -> Dict[str, float]:
        """测试实际查询性能"""
        self.logger.info("⏱️ 测试查询性能...")

        cursor = self.conn.cursor()

        # 性能测试查询
        performance_queries = [
            {
                'name': '客户总数查询',
                'query': 'SELECT COUNT(*) FROM customers'
            },
            {
                'name': '图纸总数查询',
                'query': 'SELECT COUNT(*) FROM drawings'
            },
            {
                'name': '客户图纸关联查询',
                'query': '''
                    SELECT c.company_name, COUNT(d.id) as drawing_count
                    FROM customers c
                    LEFT JOIN drawings d ON c.id = d.customer_id
                    GROUP BY c.id
                '''
            },
            {
                'name': '分类统计查询',
                'query': '''
                    SELECT product_category, COUNT(*) as count
                    FROM drawings
                    WHERE product_category IS NOT NULL
                    GROUP BY product_category
                '''
            }
        ]

        performance_results = {}

        for query_info in performance_queries:
            try:
                # 多次测试取平均值
                times = []
                for _ in range(5):
                    start_time = time.time()
                    cursor.execute(query_info['query'])
                    cursor.fetchall()
                    end_time = time.time()
                    times.append(end_time - start_time)

                avg_time = sum(times) / len(times)
                performance_results[query_info['name']] = {
                    'average_time': avg_time,
                    'min_time': min(times),
                    'max_time': max(times),
                    'query': query_info['query']
                }

            except Exception as e:
                self.logger.warning(f"性能测试失败 {query_info['name']}: {e}")
                performance_results[query_info['name']] = {'error': str(e)}

        return performance_results

    def recommend_indexes(self, table_sizes: Dict[str, int], query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """推荐索引"""
        self.logger.info("💡 生成索引建议...")

        recommendations = []

        # 分析客户表
        if table_sizes.get('customers', 0) > 10:
            # 检查是否有邮箱+公司名复合索引
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND sql LIKE '%company_name%' AND sql LIKE '%contact_email%'
            """)
            compound_index = cursor.fetchone()

            if not compound_index:
                recommendations.append({
                    'table': 'customers',
                    'columns': ['company_name', 'contact_email'],
                    'type': 'composite',
                    'reason': '核心客户标识查询优化',
                    'priority': 'high',
                    'sql': 'CREATE INDEX idx_customers_company_email_compound ON customers(company_name, contact_email)'
                })

        # 分析图纸表
        if table_sizes.get('drawings', 0) > 100:
            # 检查客户+分类复合索引
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND sql LIKE '%customer_id%' AND sql LIKE '%product_category%'
            """)
            compound_index = cursor.fetchone()

            if not compound_index:
                recommendations.append({
                    'table': 'drawings',
                    'columns': ['customer_id', 'product_category'],
                    'type': 'composite',
                    'reason': '客户产品分类查询优化',
                    'priority': 'medium',
                    'sql': 'CREATE INDEX idx_drawings_customer_category_compound ON drawings(customer_id, product_category)'
                })

            # 上传日期索引（用于时间范围查询）
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='drawings' AND sql LIKE '%upload_date%'
            """)
            date_index = cursor.fetchone()

            if not date_index:
                recommendations.append({
                    'table': 'drawings',
                    'columns': ['upload_date'],
                    'type': 'single',
                    'reason': '时间范围查询优化',
                    'priority': 'low',
                    'sql': 'CREATE INDEX idx_drawings_upload_date_optimized ON drawings(upload_date)'
                })

        # 分析报价表
        if table_sizes.get('factory_quotes', 0) > 50:
            # 工厂+日期复合索引
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND sql LIKE '%factory_id%' AND sql LIKE '%quote_date%'
            """)
            compound_index = cursor.fetchone()

            if not compound_index:
                recommendations.append({
                    'table': 'factory_quotes',
                    'columns': ['factory_id', 'quote_date'],
                    'type': 'composite',
                    'reason': '工厂报价历史查询优化',
                    'priority': 'medium',
                    'sql': 'CREATE INDEX idx_quotes_factory_date_compound ON factory_quotes(factory_id, quote_date)'
                })

        # 检查全表扫描的查询
        for query_name, analysis in query_analysis.items():
            if analysis.get('uses_index') == False:
                if 'customers' in query_name:
                    recommendations.append({
                        'table': 'customers',
                        'columns': self._extract_columns_from_query(analysis.get('query', '')),
                        'type': 'derived',
                        'reason': f'优化查询: {query_name}',
                        'priority': 'high',
                        'sql': f'-- 需要基于查询分析: {analysis.get("query", "")}'
                    })

        return recommendations

    def _extract_columns_from_query(self, query: str) -> List[str]:
        """从查询中提取WHERE子句的列"""
        # 简单的列提取逻辑
        if 'WHERE' in query.upper():
            where_clause = query.upper().split('WHERE')[1]
            columns = []
            for part in where_clause.split('AND'):
                if '=' in part:
                    column = part.split('=')[0].strip()
                    columns.append(column)
            return columns
        return []

    def create_recommended_indexes(self, recommendations: List[Dict[str, Any]], dry_run: bool = True) -> Dict[str, Any]:
        """创建推荐的索引"""
        self.logger.info(f"🔧 {'模拟' if dry_run else '执行'}索引创建...")

        results = {
            'created': [],
            'failed': [],
            'skipped': []
        }

        cursor = self.conn.cursor()

        for rec in recommendations:
            if rec.get('sql') and not rec.get('sql', '').startswith('--'):
                try:
                    if not dry_run:
                        cursor.execute(rec['sql'])
                        self.conn.commit()
                        self.logger.info(f"✅ 创建索引: {rec['sql']}")
                        results['created'].append(rec)
                    else:
                        self.logger.info(f"🔍 模拟创建: {rec['sql']}")
                        results['created'].append(rec)

                except Exception as e:
                    self.logger.error(f"❌ 索引创建失败: {rec['sql']} - {e}")
                    results['failed'].append({**rec, 'error': str(e)})
            else:
                results['skipped'].append(rec)

        return results

    def analyze_database_stats(self) -> Dict[str, Any]:
        """分析数据库统计信息"""
        self.logger.info("📈 分析数据库统计...")

        cursor = self.conn.cursor()

        stats = {
            'database_size': self._get_database_size(),
            'table_stats': self.analyze_table_sizes(),
            'index_count': len(self.get_existing_indexes()),
            'page_size': self._get_page_size(),
            'cache_size': self._get_cache_size()
        }

        return stats

    def _get_database_size(self) -> int:
        """获取数据库文件大小"""
        try:
            return Path(self.db_path).stat().st_size
        except:
            return 0

    def _get_page_size(self) -> int:
        """获取数据库页面大小"""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA page_size")
        return cursor.fetchone()[0]

    def _get_cache_size(self) -> int:
        """获取缓存大小"""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA cache_size")
        return cursor.fetchone()[0]

    def generate_optimization_report(self) -> str:
        """生成优化报告"""
        self.logger.info("📄 生成优化报告...")

        # 收集所有分析数据
        table_sizes = self.analyze_table_sizes()
        existing_indexes = self.get_existing_indexes()
        query_analysis = self.analyze_query_performance()
        performance_tests = self.test_query_performance()
        recommendations = self.recommend_indexes(table_sizes, query_analysis)
        db_stats = self.analyze_database_stats()

        # 生成报告
        report = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'database_path': self.db_path,
            'database_stats': db_stats,
            'table_sizes': table_sizes,
            'existing_indexes': existing_indexes,
            'query_analysis': query_analysis,
            'performance_tests': performance_tests,
            'recommendations': recommendations,
            'summary': {
                'total_tables': len(table_sizes),
                'total_indexes': len(existing_indexes),
                'total_recommendations': len(recommendations),
                'high_priority_recommendations': len([r for r in recommendations if r.get('priority') == 'high']),
                'medium_priority_recommendations': len([r for r in recommendations if r.get('priority') == 'medium']),
                'low_priority_recommendations': len([r for r in recommendations if r.get('priority') == 'low'])
            }
        }

        # 保存报告
        report_file = f"./reports/database_optimization_{time.strftime('%Y%m%d_%H%M%S')}.json"
        Path("./reports").mkdir(exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"📄 优化报告已保存: {report_file}")

        # 生成简要文本报告
        text_report = self._generate_text_report(report)
        text_report_file = f"./reports/database_optimization_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"

        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write(text_report)

        return report_file

    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """生成文本格式报告"""
        text = f"""
数据库优化报告
{'=' * 50}
生成时间: {report['generated_at']}
数据库路径: {report['database_path']}

📊 数据库统计:
- 数据库大小: {report['database_stats']['database_size'] / 1024:.1f} KB
- 表数量: {report['summary']['total_tables']}
- 索引数量: {report['summary']['total_indexes']}
- 页面大小: {report['database_stats']['page_size']} bytes

📋 表大小统计:
"""
        for table, count in report['table_sizes'].items():
            text += f"- {table}: {count} 条记录\n"

        text += f"""
⚡ 查询性能分析:
"""
        for query_name, analysis in report['query_analysis'].items():
            status = "✅" if analysis.get('uses_index') else "❌"
            text += f"- {status} {query_name}: {analysis.get('scan_type', 'Unknown')}\n"

        text += f"""
💡 索引建议:
- 高优先级: {report['summary']['high_priority_recommendations']} 个
- 中优先级: {report['summary']['medium_priority_recommendations']} 个
- 低优先级: {report['summary']['low_priority_recommendations']} 个

详细建议:
"""
        for rec in report['recommendations']:
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get('priority'), "⚪")
            text += f"- {priority_icon} {rec.get('table', '')}.{','.join(rec.get('columns', []))}: {rec.get('reason', '')}\n"

        return text

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

def main():
    """主函数"""
    optimizer = DatabaseOptimizer()

    try:
        # 生成优化报告
        report_file = optimizer.generate_optimization_report()

        # 获取推荐索引
        table_sizes = optimizer.analyze_table_sizes()
        query_analysis = optimizer.analyze_query_performance()
        recommendations = optimizer.recommend_indexes(table_sizes, query_analysis)

        print(f"\n🎯 数据库优化完成!")
        print(f"📄 详细报告: {report_file}")
        print(f"💡 索引建议: {len(recommendations)} 个")

        if recommendations:
            print("\n🔍 模拟创建索引...")
            results = optimizer.create_recommended_indexes(recommendations, dry_run=True)
            print(f"✅ 可创建: {len(results['created'])} 个")
            print(f"❌ 失败: {len(results['failed'])} 个")
            print(f"⏭️ 跳过: {len(results['skipped'])} 个")

    finally:
        optimizer.close()

if __name__ == "__main__":
    main()