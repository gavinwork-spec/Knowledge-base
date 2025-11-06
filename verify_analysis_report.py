#!/usr/bin/env python3
"""
验证分析报告脚本
执行分类脚本 + 分析脚本 → 检查输出结果是否合理 → 编写验证报告
"""

import sqlite3
import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import sys

class AnalysisValidator:
    """分析结果验证器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_path = db_path
        self.setup_logging()
        self.validation_results = {}

    def setup_logging(self):
        """设置日志"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'verify_analysis.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AnalysisValidator')

    def run_script_with_validation(self, script_name: str, script_args: List[str] = None) -> Dict[str, Any]:
        """运行脚本并验证结果"""
        self.logger.info(f"🚀 运行脚本: {script_name}")

        if script_args is None:
            script_args = []

        try:
            # 运行脚本
            cmd = [sys.executable, script_name] + script_args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/Users/gavin/Knowledge base"
            )

            execution_result = {
                'script': script_name,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }

            if execution_result['success']:
                self.logger.info(f"✅ {script_name} 执行成功")
            else:
                self.logger.error(f"❌ {script_name} 执行失败: {result.stderr}")

            return execution_result

        except Exception as e:
            self.logger.error(f"❌ 运行 {script_name} 时发生异常: {e}")
            return {
                'script': script_name,
                'success': False,
                'error': str(e),
                'execution_time': datetime.now().isoformat()
            }

    def validate_classification_results(self) -> Dict[str, Any]:
        """验证分类结果"""
        self.logger.info("🔍 验证分类结果...")

        try:
            conn = sqlite3.connect(self.db_path)

            # 1. 检查分类覆盖率
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM drawings")
            total_drawings = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM drawings WHERE is_classified = 1")
            classified_drawings = cursor.fetchone()[0]

            classification_rate = (classified_drawings / total_drawings * 100) if total_drawings > 0 else 0

            # 2. 检查分类分布
            cursor.execute("""
                SELECT product_category, COUNT(*) as count
                FROM drawings
                GROUP BY product_category
                ORDER BY count DESC
            """)
            category_distribution = cursor.fetchall()

            # 3. 检查标准件vs定制件分布
            cursor.execute("""
                SELECT
                    standard_or_custom,
                    COUNT(*) as count,
                    COUNT(DISTINCT product_category) as categories
                FROM drawings
                WHERE standard_or_custom IS NOT NULL
                GROUP BY standard_or_custom
            """)
            standard_custom_dist = [(row[0], row[1]) for row in cursor.fetchall()]  # 只取前两列

            # 4. 检查置信度分布
            cursor.execute("""
                SELECT
                    CASE
                        WHEN classification_confidence >= 0.8 THEN '高'
                        WHEN classification_confidence >= 0.5 THEN '中'
                        WHEN classification_confidence > 0 THEN '低'
                        ELSE '无'
                    END as confidence_level,
                    COUNT(*) as count
                FROM drawings
                GROUP BY confidence_level
            """)
            confidence_distribution = cursor.fetchall()

            # 5. 检查数据源分布
            cursor.execute("""
                SELECT data_source, COUNT(*) as count
                FROM drawings
                WHERE data_source IS NOT NULL AND data_source != 'unknown'
                GROUP BY data_source
                ORDER BY count DESC
            """)
            source_distribution = cursor.fetchall()

            conn.close()

            # 验证规则
            validation_rules = {
                'coverage_rate_acceptable': classification_rate >= 5,  # 至少5%分类率
                'has_multiple_categories': len(category_distribution) >= 2,
                'has_custom_items': any(row[0] == 1 for row in standard_custom_dist),  # 有定制件
                'has_confidence_scores': any(row[0] != '无' for row in confidence_distribution),
                'reasonable_distribution': self._validate_category_distribution(category_distribution)
            }

            validation_result = {
                'classification_metrics': {
                    'total_drawings': total_drawings,
                    'classified_drawings': classified_drawings,
                    'classification_rate': round(classification_rate, 2),
                    'category_distribution': {cat: count for cat, count in category_distribution},
                    'standard_custom_distribution': {std: count for std, count in standard_custom_dist},
                    'confidence_distribution': {conf: count for conf, count in confidence_distribution},
                    'source_distribution': {src: count for src, count in source_distribution}
                },
                'validation_rules': validation_rules,
                'overall_status': 'PASS' if all(validation_rules.values()) else 'NEEDS_ATTENTION'
            }

            self.logger.info(f"✅ 分类验证完成: {validation_result['overall_status']}")
            return validation_result

        except Exception as e:
            self.logger.error(f"❌ 分类验证失败: {e}")
            return {'error': str(e), 'overall_status': 'ERROR'}

    def _validate_category_distribution(self, distribution: List[Tuple]) -> bool:
        """验证分类分布是否合理"""
        if not distribution:
            return False

        # 检查是否有主导分类
        total_count = sum(count for _, count in distribution)
        if total_count == 0:
            return False

        # 主导分类不应超过90%
        max_count = max(count for _, count in distribution)
        if max_count / total_count > 0.9:
            return False

        # 至少有一个合理的分类（不是未分类）
        reasonable_categories = sum(1 for category, count in distribution
                                  if category != '未分类' and count > 0)

        return reasonable_categories >= 1

    def validate_analysis_results(self) -> Dict[str, Any]:
        """验证分析结果"""
        self.logger.info("🔍 验证分析结果...")

        try:
            # 检查分析输出文件
            processed_dir = Path("./data/processed")
            analysis_files = list(processed_dir.glob("*analysis*"))
            trend_files = list(processed_dir.glob("*trends*"))
            statistics_files = list(processed_dir.glob("*statistics*"))

            validation_result = {
                'output_files': {
                    'analysis_files': len(analysis_files),
                    'trend_files': len(trend_files),
                    'statistics_files': len(statistics_files),
                    'total_files': len(analysis_files) + len(trend_files) + len(statistics_files)
                },
                'file_validation': self._validate_output_files(analysis_files + trend_files + statistics_files)
            }

            # 验证数据库中的分析数据
            db_validation = self._validate_database_analysis_data()
            validation_result.update(db_validation)

            self.logger.info(f"✅ 分析验证完成: {validation_result.get('overall_status', 'UNKNOWN')}")
            return validation_result

        except Exception as e:
            self.logger.error(f"❌ 分析验证失败: {e}")
            return {'error': str(e), 'overall_status': 'ERROR'}

    def _validate_output_files(self, file_list: List[Path]) -> Dict[str, Any]:
        """验证输出文件"""
        file_validation = {
            'total_files': len(file_list),
            'valid_files': 0,
            'invalid_files': 0,
            'file_details': []
        }

        for file_path in file_list:
            try:
                if file_path.suffix == '.json':
                    # 验证JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    file_validation['valid_files'] += 1
                    file_validation['file_details'].append({
                        'file': file_path.name,
                        'type': 'json',
                        'size': file_path.stat().st_size,
                        'valid': True
                    })

                elif file_path.suffix == '.csv':
                    # 验证CSV文件
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    if len(df) > 0:
                        file_validation['valid_files'] += 1
                        file_validation['file_details'].append({
                            'file': file_path.name,
                            'type': 'csv',
                            'rows': len(df),
                            'columns': len(df.columns),
                            'valid': True
                        })
                    else:
                        file_validation['invalid_files'] += 1
                        file_validation['file_details'].append({
                            'file': file_path.name,
                            'type': 'csv',
                            'valid': False,
                            'issue': 'Empty file'
                        })
                else:
                    file_validation['invalid_files'] += 1

            except Exception as e:
                file_validation['invalid_files'] += 1
                file_validation['file_details'].append({
                    'file': file_path.name,
                    'valid': False,
                    'issue': str(e)
                })

        return file_validation

    def _validate_database_analysis_data(self) -> Dict[str, Any]:
        """验证数据库中的分析数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            validation_data = {}

            # 验证工厂报价时间字段
            cursor.execute("""
                SELECT COUNT(*) FROM factory_quotes
                WHERE quote_month IS NOT NULL AND quote_year IS NOT NULL
            """)
            quotes_with_time_fields = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM factory_quotes")
            total_quotes = cursor.fetchone()[0]

            validation_data['quote_time_fields'] = {
                'total_quotes': total_quotes,
                'quotes_with_time_fields': quotes_with_time_fields,
                'completion_rate': (quotes_with_time_fields / total_quotes * 100) if total_quotes > 0 else 0
            }

            # 验证图纸分类字段
            cursor.execute("""
                SELECT COUNT(*) FROM drawings
                WHERE is_classified = 1 AND classification_date IS NOT NULL
            """)
            drawings_with_classification = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM drawings")
            total_drawings = cursor.fetchone()[0]

            validation_data['drawing_classification'] = {
                'total_drawings': total_drawings,
                'drawings_with_classification': drawings_with_classification,
                'classification_rate': (drawings_with_classification / total_drawings * 100) if total_drawings > 0 else 0
            }

            # 验证客户统计字段
            cursor.execute("""
                SELECT COUNT(*) FROM customers
                WHERE total_drawings >= 0
            """)
            customers_with_stats = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM customers")
            total_customers = cursor.fetchone()[0]

            validation_data['customer_statistics'] = {
                'total_customers': total_customers,
                'customers_with_stats': customers_with_stats,
                'stats_completion_rate': (customers_with_stats / total_customers * 100) if total_customers > 0 else 0
            }

            conn.close()

            # 整体状态
            overall_scores = [
                validation_data['quote_time_fields']['completion_rate'],
                validation_data['drawing_classification']['classification_rate'],
                validation_data['customer_statistics']['stats_completion_rate']
            ]

            validation_data['overall_analysis_quality'] = sum(overall_scores) / len(overall_scores)
            validation_data['overall_status'] = 'GOOD' if validation_data['overall_analysis_quality'] >= 80 else 'NEEDS_IMPROVEMENT'

            return validation_data

        except Exception as e:
            self.logger.error(f"❌ 数据库分析数据验证失败: {e}")
            return {'error': str(e), 'overall_status': 'ERROR'}

    def validate_data_quality(self) -> Dict[str, Any]:
        """验证数据质量"""
        self.logger.info("🔍 验证数据质量...")

        try:
            # 运行数据质量检查脚本
            quality_result = self.run_script_with_validation('data_quality_check.py')

            if quality_result['success']:
                # 解析质量检查结果
                quality_output = quality_result['stdout']

                quality_validation = {
                    'script_execution': 'SUCCESS',
                    'quality_issues_detected': '❌' in quality_output,
                    'data_quality_grade': self._extract_quality_grade(quality_output),
                    'recommendations': self._extract_recommendations(quality_output)
                }
            else:
                quality_validation = {
                    'script_execution': 'FAILED',
                    'error': quality_result.get('stderr', 'Unknown error')
                }

            self.logger.info(f"✅ 数据质量验证完成: {quality_validation.get('script_execution', 'UNKNOWN')}")
            return quality_validation

        except Exception as e:
            self.logger.error(f"❌ 数据质量验证失败: {e}")
            return {'error': str(e), 'script_execution': 'ERROR'}

    def _extract_quality_grade(self, output: str) -> str:
        """从输出中提取数据质量等级"""
        if '数据质量等级: A' in output:
            return 'A'
        elif '数据质量等级: B' in output:
            return 'B'
        elif '数据质量等级: C' in output:
            return 'C'
        elif '数据质量等级: D' in output:
            return 'D'
        else:
            return 'UNKNOWN'

    def _extract_recommendations(self, output: str) -> List[str]:
        """从输出中提取建议"""
        recommendations = []
        lines = output.split('\n')

        for line in lines:
            if line.strip().startswith('- '):
                recommendations.append(line.strip())

        return recommendations

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """生成验证报告"""
        self.logger.info("📄 生成验证报告...")

        try:
            report_dir = Path("./data/processed")
            report_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"validation_report_{timestamp}.json"

            # 生成综合报告
            comprehensive_report = {
                'validation_timestamp': datetime.now().isoformat(),
                'execution_summary': {
                    'total_scripts_executed': len([r for r in results.get('script_executions', []) if r.get('success')]),
                    'total_scripts_failed': len([r for r in results.get('script_executions', []) if not r.get('success')]),
                    'overall_success_rate': len([r for r in results.get('script_executions', []) if r.get('success')]) / len(results.get('script_executions', [])) * 100 if results.get('script_executions') else 0
                },
                'classification_validation': results.get('classification_validation', {}),
                'analysis_validation': results.get('analysis_validation', {}),
                'data_quality_validation': results.get('data_quality_validation', {}),
                'overall_assessment': self._generate_overall_assessment(results),
                'recommendations': self._generate_recommendations(results),
                'next_steps': self._generate_next_steps(results)
            }

            # 保存JSON报告
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)

            # 生成文本报告
            text_report = self._generate_text_report(comprehensive_report)
            text_report_file = report_dir / f"validation_report_summary_{timestamp}.txt"

            with open(text_report_file, 'w', encoding='utf-8') as f:
                f.write(text_report)

            self.logger.info(f"✅ 验证报告已保存: {report_file}")
            return str(report_file)

        except Exception as e:
            self.logger.error(f"❌ 生成验证报告失败: {e}")
            return ""

    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成总体评估"""
        assessment = {
            'overall_status': 'UNKNOWN',
            'classification_quality': 'UNKNOWN',
            'analysis_quality': 'UNKNOWN',
            'data_quality': 'UNKNOWN',
            'system_health': 'UNKNOWN'
        }

        # 分类质量评估
        classification_val = results.get('classification_validation', {})
        if classification_val.get('overall_status') == 'PASS':
            assessment['classification_quality'] = 'GOOD'
        elif classification_val.get('overall_status') == 'NEEDS_ATTENTION':
            assessment['classification_quality'] = 'FAIR'
        else:
            assessment['classification_quality'] = 'POOR'

        # 分析质量评估
        analysis_val = results.get('analysis_validation', {})
        if analysis_val.get('overall_status') == 'GOOD':
            assessment['analysis_quality'] = 'GOOD'
        elif analysis_val.get('overall_status') == 'NEEDS_IMPROVEMENT':
            assessment['analysis_quality'] = 'FAIR'
        else:
            assessment['analysis_quality'] = 'POOR'

        # 数据质量评估
        data_quality_val = results.get('data_quality_validation', {})
        if data_quality_val.get('script_execution') == 'SUCCESS':
            grade = data_quality_val.get('data_quality_grade', 'UNKNOWN')
            if grade in ['A', 'B']:
                assessment['data_quality'] = 'GOOD'
            elif grade == 'C':
                assessment['data_quality'] = 'FAIR'
            else:
                assessment['data_quality'] = 'POOR'
        else:
            assessment['data_quality'] = 'POOR'

        # 系统健康评估
        scores = []
        if assessment['classification_quality'] == 'GOOD':
            scores.append(1)
        elif assessment['classification_quality'] == 'FAIR':
            scores.append(0.5)
        else:
            scores.append(0)

        if assessment['analysis_quality'] == 'GOOD':
            scores.append(1)
        elif assessment['analysis_quality'] == 'FAIR':
            scores.append(0.5)
        else:
            scores.append(0)

        if assessment['data_quality'] == 'GOOD':
            scores.append(1)
        elif assessment['data_quality'] == 'FAIR':
            scores.append(0.5)
        else:
            scores.append(0)

        overall_score = sum(scores) / len(scores) if scores else 0

        if overall_score >= 0.8:
            assessment['overall_status'] = 'EXCELLENT'
            assessment['system_health'] = 'HEALTHY'
        elif overall_score >= 0.6:
            assessment['overall_status'] = 'GOOD'
            assessment['system_health'] = 'HEALTHY'
        elif overall_score >= 0.4:
            assessment['overall_status'] = 'FAIR'
            assessment['system_health'] = 'NEEDS_ATTENTION'
        else:
            assessment['overall_status'] = 'POOR'
            assessment['system_health'] = 'CRITICAL'

        assessment['overall_score'] = round(overall_score * 100, 2)

        return assessment

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 分类改进建议
        classification_val = results.get('classification_validation', {})
        if classification_val.get('overall_status') != 'PASS':
            recommendations.append("改进分类规则，提高分类覆盖率")
            recommendations.append("扩充关键词库，增强分类准确性")

        # 分析改进建议
        analysis_val = results.get('analysis_validation', {})
        if analysis_val.get('overall_status') != 'GOOD':
            recommendations.append("完善分析脚本，确保数据完整性")
            recommendations.append("增加更多维度的分析指标")

        # 数据质量改进建议
        data_quality_val = results.get('data_quality_validation', {})
        if data_quality_val.get('script_execution') != 'SUCCESS':
            recommendations.append("修复数据质量检查脚本")
        else:
            recommendations.extend(data_quality_val.get('recommendations', []))

        # 通用建议
        recommendations.append("定期执行验证流程，监控系统健康状态")
        recommendations.append("建立自动化监控和预警机制")

        return recommendations

    def _generate_next_steps(self, results: Dict[str, Any]) -> List[str]:
        """生成下一步行动计划"""
        next_steps = []

        assessment = self._generate_overall_assessment(results)

        if assessment['overall_status'] in ['EXCELLENT', 'GOOD']:
            next_steps.append("系统运行良好，继续按计划执行定期维护")
            next_steps.append("准备进入下一阶段开发（API接口设计）")
        elif assessment['overall_status'] == 'FAIR':
            next_steps.append("优先解决数据质量问题")
            next_steps.append("优化分类算法，提高准确率")
            next_steps.append("完善分析脚本功能")
        else:
            next_steps.append("立即修复关键问题")
            next_steps.append("重新设计数据流程")
            next_steps.append("加强测试和验证机制")

        return next_steps

    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """生成文本格式报告"""
        text = f"""
知识库系统验证报告
{'=' * 50}
生成时间: {report['validation_timestamp']}

总体评估
--------
系统状态: {report['overall_assessment']['overall_status']}
健康评分: {report['overall_assessment']['overall_score']}/100
分类质量: {report['overall_assessment']['classification_quality']}
分析质量: {report['overall_assessment']['analysis_quality']}
数据质量: {report['overall_assessment']['data_quality']}

执行摘要
--------
脚本执行成功率: {report['execution_summary']['overall_success_rate']:.1f}%
成功执行: {report['execution_summary']['total_scripts_executed']} 个
执行失败: {report['execution_summary']['total_scripts_failed']} 个

分类验证结果
--------
分类状态: {report['classification_validation'].get('overall_status', 'UNKNOWN')}
分类覆盖率: {report['classification_validation'].get('classification_metrics', {}).get('classification_rate', 0):.1f}%

分析验证结果
--------
分析状态: {report['analysis_validation'].get('overall_status', 'UNKNOWN')}
输出文件数: {report['analysis_validation'].get('output_files', {}).get('total_files', 0)}

数据质量验证
--------
验证状态: {report['data_quality_validation'].get('script_execution', 'UNKNOWN')}
质量等级: {report['data_quality_validation'].get('data_quality_grade', 'UNKNOWN')}

改进建议
--------
{chr(10).join(f"- {rec}" for rec in report.get('recommendations', []))}

下一步行动
--------
{chr(10).join(f"- {step}" for step in report.get('next_steps', []))}

{'=' * 50}
报告生成完成
"""
        return text

    def run_full_validation(self) -> Dict[str, Any]:
        """运行完整验证流程"""
        self.logger.info("🚀 开始完整验证流程...")

        start_time = datetime.now()

        try:
            # 1. 执行分类脚本
            self.logger.info("📝 执行分类脚本...")
            classification_result = self.run_script_with_validation('classify_drawings.py', ['--report'])

            # 2. 执行分析脚本
            self.logger.info("📊 执行分析脚本...")
            analysis_result = self.run_script_with_validation('analyze_factory_quote_trends.py')

            # 3. 执行统计导出脚本
            self.logger.info("💾 执行统计导出脚本...")
            statistics_result = self.run_script_with_validation('export_statistics.py')

            # 4. 验证结果
            self.logger.info("🔍 验证执行结果...")
            classification_validation = self.validate_classification_results()
            analysis_validation = self.validate_analysis_results()
            data_quality_validation = self.validate_data_quality()

            # 5. 生成报告
            self.logger.info("📄 生成验证报告...")
            validation_results = {
                'script_executions': {
                    'classification': classification_result,
                    'analysis': analysis_result,
                    'statistics': statistics_result
                },
                'classification_validation': classification_validation,
                'analysis_validation': analysis_validation,
                'data_quality_validation': data_quality_validation
            }

            report_file = self.generate_validation_report(validation_results)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            final_result = {
                'success': True,
                'processing_time': processing_time,
                'validation_report': report_file,
                'overall_assessment': self._generate_overall_assessment(validation_results),
                'validation_results': validation_results
            }

            self.logger.info(f"✅ 完整验证流程完成! 耗时: {processing_time:.2f}秒")
            self.logger.info(f"📄 验证报告: {report_file}")

            return final_result

        except Exception as e:
            self.logger.error(f"❌ 验证流程失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='分析结果验证工具')
    parser.add_argument('--db-path', default='./data/db.sqlite', help='数据库文件路径')
    parser.add_argument('--component', choices=['classification', 'analysis', 'quality', 'all'], default='all', help='验证组件')
    parser.add_argument('--skip-execution', action='store_true', help='跳过脚本执行，仅验证现有结果')

    args = parser.parse_args()

    validator = AnalysisValidator(args.db_path)

    if args.skip_execution:
        # 仅验证现有结果
        classification_validation = validator.validate_classification_results()
        analysis_validation = validator.validate_analysis_results()
        data_quality_validation = validator.validate_data_quality()

        results = {
            'classification_validation': classification_validation,
            'analysis_validation': analysis_validation,
            'data_quality_validation': data_quality_validation
        }

        report_file = validator.generate_validation_report(results)
        print(f"✅ 验证完成! 报告: {report_file}")

    else:
        # 运行完整验证流程
        result = validator.run_full_validation()

        if result['success']:
            print("✅ 分析验证完成!")
            print(f"📊 总体评估: {result['overall_assessment']['overall_status']}")
            print(f"💯 健康评分: {result['overall_assessment']['overall_score']}/100")
            print(f"⏱️ 处理时间: {result['processing_time']:.2f}秒")
            print(f"📄 验证报告: {result['validation_report']}")
        else:
            print(f"❌ 验证失败: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    main()