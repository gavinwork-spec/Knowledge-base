#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge System Verification
知识库系统验证脚本

This script performs comprehensive verification of all knowledge base system components
and generates a detailed health report.
"""

import sqlite3
import json
import logging
import requests
import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/system_verification.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemVerificationManager:
    """系统验证管理器"""

    def __init__(self):
        self.db_path = "knowledge_base.db"
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_status': 'unknown',
            'issues': [],
            'recommendations': []
        }

    def verify_database_schema(self) -> Dict[str, Any]:
        """验证数据库架构"""
        result = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 检查必需的表
            required_tables = [
                'knowledge_entries',
                'entity_types',
                'nlp_entities',
                'strategy_suggestions',
                'embedding_index',
                'knowledge_relationships'
            ]

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]

            missing_tables = [table for table in required_tables if table not in existing_tables]

            if missing_tables:
                result['status'] = 'error'
                result['issues'].append(f"Missing tables: {', '.join(missing_tables)}")
            else:
                result['status'] = 'pass'
                result['details']['tables_count'] = len(existing_tables)

            # 检查索引
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
            indexes = [row[0] for row in cursor.fetchall()]
            result['details']['indexes_count'] = len(indexes)

            # 检查视图
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
            views = [row[0] for row in cursor.fetchall()]
            result['details']['views_count'] = len(views)

            # 检查触发器
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
            triggers = [row[0] for row in cursor.fetchall()]
            result['details']['triggers_count'] = len(triggers)

            # 检查数据完整性
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            entry_count = cursor.fetchone()[0]
            result['details']['knowledge_entries_count'] = entry_count

            cursor.execute("SELECT COUNT(*) FROM entity_types")
            type_count = cursor.fetchone()[0]
            result['details']['entity_types_count'] = type_count

            conn.close()

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"Database verification failed: {str(e)}")

        return result

    def verify_knowledge_models_setup(self) -> Dict[str, Any]:
        """验证知识模型设置"""
        result = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }

        try:
            # 检查设置脚本是否存在
            setup_script = Path("setup_knowledge_models.py")
            if not setup_script.exists():
                result['status'] = 'error'
                result['issues'].append("setup_knowledge_models.py not found")
                return result

            # 尝试导入模块
            try:
                sys.path.insert(0, '.')
                import setup_knowledge_models
                result['status'] = 'pass'
                result['details']['module_imported'] = True
            except ImportError as e:
                result['status'] = 'error'
                result['issues'].append(f"Failed to import setup module: {str(e)}")
                return result

            # 验证实体类型数据
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM entity_types WHERE is_active = 1")
            active_types = cursor.fetchone()[0]
            result['details']['active_entity_types'] = active_types

            if active_types < 5:
                result['issues'].append(f"Low number of active entity types: {active_types}")

            # 验证示例数据
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            entries_count = cursor.fetchone()[0]
            result['details']['sample_entries'] = entries_count

            conn.close()

            if result['status'] == 'pass' and not result['issues']:
                result['status'] = 'pass'

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"Knowledge models verification failed: {str(e)}")

        return result

    def verify_document_parsing(self) -> Dict[str, Any]:
        """验证文档解析功能"""
        result = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }

        try:
            # 检查解析脚本
            parse_script = Path("parse_documents.py")
            if not parse_script.exists():
                result['status'] = 'error'
                result['issues'].append("parse_documents.py not found")
                return result

            # 检查代理配置
            agent_config = Path("parse_documents_agent.yaml")
            if not agent_config.exists():
                result['status'] = 'error'
                result['issues'].append("parse_documents_agent.yaml not found")
                return result

            result['details']['parsing_script_exists'] = True
            result['details']['agent_config_exists'] = True

            # 尝试导入解析模块
            try:
                import parse_documents
                result['details']['module_imported'] = True
            except ImportError as e:
                result['issues'].append(f"Failed to import parse module: {str(e)}")

            # 检查支持的文件格式
            try:
                import yaml
                with open(agent_config, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                supported_formats = config.get('config', {}).get('parsing', {}).get('supported_formats', [])
                result['details']['supported_formats'] = supported_formats
                result['details']['supported_formats_count'] = len(supported_formats)

            except Exception as e:
                result['issues'].append(f"Failed to read agent config: {str(e)}")

            # 检查NLP实体数据
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM nlp_entities")
            nlp_count = cursor.fetchone()[0]
            result['details']['nlp_entities_count'] = nlp_count

            conn.close()

            if result['issues']:
                result['status'] = 'warning'
            else:
                result['status'] = 'pass'

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"Document parsing verification failed: {str(e)}")

        return result

    def verify_embedding_index(self) -> Dict[str, Any]:
        """验证嵌入索引"""
        result = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }

        try:
            # 检查嵌入脚本
            embedding_script = Path("build_embeddings.py")
            if not embedding_script.exists():
                result['status'] = 'error'
                result['issues'].append("build_embeddings.py not found")
                return result

            result['details']['embedding_script_exists'] = True

            # 尝试导入嵌入模块
            try:
                import build_embeddings
                result['details']['module_imported'] = True
            except ImportError as e:
                result['issues'].append(f"Failed to import embedding module: {str(e)}")

            # 检查嵌入索引数据
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM embedding_index")
            embedding_count = cursor.fetchone()[0]
            result['details']['embedding_count'] = embedding_count

            # 检查模型信息
            cursor.execute("SELECT DISTINCT model_name FROM embedding_index")
            models = [row[0] for row in cursor.fetchall()]
            result['details']['available_models'] = models

            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            total_entries = cursor.fetchone()[0]

            if total_entries > 0:
                coverage = (embedding_count / total_entries) * 100
                result['details']['embedding_coverage'] = f"{coverage:.1f}%"

                if coverage < 50:
                    result['issues'].append(f"Low embedding coverage: {coverage:.1f}%")
                elif coverage < 90:
                    result['issues'].append(f"Moderate embedding coverage: {coverage:.1f}%")
            else:
                result['issues'].append("No knowledge entries found")

            conn.close()

            if result['issues']:
                result['status'] = 'warning'
            else:
                result['status'] = 'pass'

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"Embedding index verification failed: {str(e)}")

        return result

    def verify_knowledge_api(self) -> Dict[str, Any]:
        """验证知识库API"""
        result = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }

        try:
            api_host = "localhost"
            api_port = 8001
            base_url = f"http://{api_host}:{api_port}"

            # 检查API脚本
            api_script = Path("api_server_knowledge.py")
            if not api_script.exists():
                result['status'] = 'error'
                result['issues'].append("api_server_knowledge.py not found")
                return result

            result['details']['api_script_exists'] = True

            # 测试健康检查端点
            try:
                response = requests.get(f"{base_url}/api/health", timeout=5)
                if response.status_code == 200:
                    result['status'] = 'pass'
                    result['details']['health_check'] = 'pass'
                    health_data = response.json()
                    result['details']['api_version'] = health_data.get('version', 'unknown')
                else:
                    result['status'] = 'error'
                    result['issues'].append(f"Health check failed: HTTP {response.status_code}")
            except requests.exceptions.ConnectionError:
                result['status'] = 'error'
                result['issues'].append("Cannot connect to API server - is it running?")
                return result
            except requests.exceptions.Timeout:
                result['status'] = 'error'
                result['issues'].append("API server timeout")
                return result

            # 测试知识条目端点
            try:
                response = requests.get(f"{base_url}/api/knowledge/entries?limit=5", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    result['details']['entries_endpoint'] = 'pass'
                    result['details']['entries_count'] = len(data.get('data', {}).get('entries', []))
                else:
                    result['issues'].append(f"Entries endpoint failed: HTTP {response.status_code}")
            except Exception as e:
                result['issues'].append(f"Entries endpoint error: {str(e)}")

            # 测试搜索端点
            try:
                response = requests.get(f"{base_url}/api/knowledge/search?q=test&top_k=3", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    result['details']['search_endpoint'] = 'pass'
                    result['details']['search_results_count'] = len(data.get('data', {}).get('results', []))
                else:
                    result['issues'].append(f"Search endpoint failed: HTTP {response.status_code}")
            except Exception as e:
                result['issues'].append(f"Search endpoint error: {str(e)}")

            # 测试统计端点
            try:
                response = requests.get(f"{base_url}/api/knowledge/stats", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    result['details']['stats_endpoint'] = 'pass'
                    result['details']['total_entries'] = data.get('data', {}).get('total_entries', 0)
                else:
                    result['issues'].append(f"Stats endpoint failed: HTTP {response.status_code}")
            except Exception as e:
                result['issues'].append(f"Stats endpoint error: {str(e)}")

            if result['issues']:
                result['status'] = 'warning'

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"API verification failed: {str(e)}")

        return result

    def verify_quote_analysis(self) -> Dict[str, Any]:
        """验证报价分析功能"""
        result = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }

        try:
            # 检查分析脚本
            analysis_script = Path("quote_analysis_agent.py")
            if not analysis_script.exists():
                result['status'] = 'error'
                result['issues'].append("quote_analysis_agent.py not found")
                return result

            result['details']['analysis_script_exists'] = True

            # 尝试导入分析模块
            try:
                import quote_analysis_agent
                result['details']['module_imported'] = True
            except ImportError as e:
                result['issues'].append(f"Failed to import analysis module: {str(e)}")

            # 检查策略建议数据
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM strategy_suggestions
                WHERE created_at >= date('now', '-7 days')
            """)
            recent_suggestions = cursor.fetchone()[0]
            result['details']['recent_suggestions'] = recent_suggestions

            cursor.execute("""
                SELECT suggestion_type, COUNT(*) as count
                FROM strategy_suggestions
                GROUP BY suggestion_type
            """)
            suggestions_by_type = dict(cursor.fetchall())
            result['details']['suggestions_by_type'] = suggestions_by_type

            conn.close()

            if result['issues']:
                result['status'] = 'warning'
            else:
                result['status'] = 'pass'

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"Quote analysis verification failed: {str(e)}")

        return result

    def verify_github_migration(self) -> Dict[str, Any]:
        """验证GitHub迁移功能"""
        result = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }

        try:
            # 检查迁移脚本
            migration_script = Path("github_migration.py")
            if not migration_script.exists():
                result['status'] = 'error'
                result['issues'].append("github_migration.py not found")
                return result

            result['details']['migration_script_exists'] = True

            # 尝试导入迁移模块
            try:
                import github_migration
                result['details']['module_imported'] = True
            except ImportError as e:
                result['issues'].append(f"Failed to import migration module: {str(e)}")

            # 检查备份仓库
            backup_repo = Path("knowledge-github-backup")
            if backup_repo.exists():
                result['details']['backup_repo_exists'] = True

                # 检查仓库文件
                repo_files = list(backup_repo.rglob('*'))
                result['details']['backup_repo_files'] = len([f for f in repo_files if f.is_file()])

                # 检查必要文件
                required_files = ['README.md', 'MIGRATION_REPORT.md', 'knowledge_data.json']
                missing_files = [f for f in required_files if not (backup_repo / f).exists()]

                if missing_files:
                    result['issues'].append(f"Missing backup files: {', '.join(missing_files)}")

                # 检查Git仓库
                if (backup_repo / '.git').exists():
                    result['details']['git_repo_initialized'] = True
                else:
                    result['issues'].append("Git repository not initialized in backup")
            else:
                result['issues'].append("Backup repository not found - run migration first")

            if result['issues']:
                result['status'] = 'warning'
            else:
                result['status'] = 'pass'

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"GitHub migration verification failed: {str(e)}")

        return result

    def verify_frontend_interface(self) -> Dict[str, Any]:
        """验证前端界面"""
        result = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }

        try:
            # 检查前端文件
            frontend_file = Path("github-frontend/knowledge.html")
            if not frontend_file.exists():
                result['status'] = 'error'
                result['issues'].append("knowledge.html frontend not found")
                return result

            result['details']['frontend_exists'] = True
            result['details']['frontend_size'] = frontend_file.stat().st_size

            # 检查前端内容
            try:
                with open(frontend_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查关键组件
                required_elements = [
                    'Knowledge Hub',
                    'apiClient',
                    'performSearch',
                    'bootstrap'
                ]

                missing_elements = [elem for elem in required_elements if elem not in content]

                if missing_elements:
                    result['issues'].append(f"Missing frontend elements: {', '.join(missing_elements)}")

            except Exception as e:
                result['issues'].append(f"Failed to read frontend file: {str(e)}")

            # 检查API客户端文件
            api_client = Path("github-frontend/assets/js/api.js")
            if api_client.exists():
                result['details']['api_client_exists'] = True
            else:
                result['issues'].append("API client file not found")

            if result['issues']:
                result['status'] = 'warning'
            else:
                result['status'] = 'pass'

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"Frontend verification failed: {str(e)}")

        return result

    def run_full_verification(self) -> Dict[str, Any]:
        """运行完整系统验证"""
        logger.info("🔍 Starting comprehensive system verification...")

        verifications = [
            ('database_schema', self.verify_database_schema),
            ('knowledge_models', self.verify_knowledge_models_setup),
            ('document_parsing', self.verify_document_parsing),
            ('embedding_index', self.verify_embedding_index),
            ('knowledge_api', self.verify_knowledge_api),
            ('quote_analysis', self.verify_quote_analysis),
            ('github_migration', self.verify_github_migration),
            ('frontend_interface', self.verify_frontend_interface)
        ]

        for component_name, verify_func in verifications:
            logger.info(f"Verifying {component_name}...")
            result = verify_func()
            self.verification_results['components'][component_name] = result

            if result['status'] == 'error':
                self.verification_results['issues'].extend(result['issues'])
            elif result['status'] == 'warning':
                self.verification_results['recommendations'].extend(result['issues'])

        # 计算总体状态
        statuses = [comp['status'] for comp in self.verification_results['components'].values()]

        if 'error' in statuses:
            self.verification_results['overall_status'] = 'error'
        elif 'warning' in statuses:
            self.verification_results['overall_status'] = 'warning'
        elif all(status == 'pass' for status in statuses):
            self.verification_results['overall_status'] = 'pass'
        else:
            self.verification_results['overall_status'] = 'unknown'

        logger.info(f"✅ System verification completed. Overall status: {self.verification_results['overall_status']}")

        return self.verification_results

    def generate_report(self, output_file: str = None) -> str:
        """生成验证报告"""
        if not output_file:
            output_file = f"system_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        # 计算统计信息
        total_components = len(self.verification_results['components'])
        passed_components = sum(1 for comp in self.verification_results['components'].values() if comp['status'] == 'pass')
        warning_components = sum(1 for comp in self.verification_results['components'].values() if comp['status'] == 'warning')
        error_components = sum(1 for comp in self.verification_results['components'].values() if comp['status'] == 'error')

        # 生成报告
        report = f"""# Knowledge Base System Verification Report

**Verification Time**: {self.verification_results['timestamp']}
**Overall Status**: {self.verification_results['overall_status'].upper()}

## Summary

- **Total Components**: {total_components}
- **Passed**: {passed_components} ✅
- **Warnings**: {warning_components} ⚠️
- **Errors**: {error_components} ❌

## Component Details

"""

        for component_name, result in self.verification_results['components'].items():
            status_icon = {'pass': '✅', 'warning': '⚠️', 'error': '❌', 'unknown': '❓'}.get(result['status'], '❓')

            report += f"### {component_name.replace('_', ' ').title()} {status_icon}\n\n"
            report += f"**Status**: {result['status']}\n\n"

            if result['details']:
                report += "**Details**:\n"
                for key, value in result['details'].items():
                    report += f"- {key.replace('_', ' ').title()}: {value}\n"
                report += "\n"

            if result['issues']:
                report += "**Issues**:\n"
                for issue in result['issues']:
                    report += f"- ❌ {issue}\n"
                report += "\n"

        if self.verification_results['recommendations']:
            report += "## Recommendations\n\n"
            for rec in self.verification_results['recommendations']:
                report += f"- 💡 {rec}\n"
            report += "\n"

        if self.verification_results['issues']:
            report += "## Critical Issues\n\n"
            for issue in self.verification_results['issues']:
                report += f"- 🚨 {issue}\n"
            report += "\n"

        report += f"""
## System Health Score

**Score: {(passed_components / total_components) * 100:.1f}%**

{self._get_health_recommendation(passed_components, total_components)}

---

*Report generated by Knowledge Base System Verification Manager v1.0.0*
"""

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"📋 Verification report saved to: {output_file}")
        return output_file

    def _get_health_recommendation(self, passed: int, total: int) -> str:
        """获取健康建议"""
        percentage = (passed / total) * 100

        if percentage >= 90:
            return "🟢 System is in excellent condition!"
        elif percentage >= 75:
            return "🟡 System is functional but has some issues to address."
        elif percentage >= 50:
            return "🟠 System has significant issues that need attention."
        else:
            return "🔴 System requires immediate attention and fixes."

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Knowledge System Verification')
    parser.add_argument('--verify-all', action='store_true', help='Run full system verification')
    parser.add_argument('--component', choices=[
        'database_schema', 'knowledge_models', 'document_parsing',
        'embedding_index', 'knowledge_api', 'quote_analysis',
        'github_migration', 'frontend_interface'
    ], help='Verify specific component')
    parser.add_argument('--report', help='Output report file path')
    parser.add_argument('--quick', action='store_true', help='Quick verification (skip API tests)')

    args = parser.parse_args()

    verifier = SystemVerificationManager()

    if args.verify_all:
        results = verifier.run_full_verification()

        print(f"\n🔍 System Verification Results")
        print("=" * 50)
        print(f"Overall Status: {results['overall_status'].upper()}")

        for component, result in results['components'].items():
            status_icon = {'pass': '✅', 'warning': '⚠️', 'error': '❌', 'unknown': '❓'}.get(result['status'], '❓')
            print(f"  {component.replace('_', ' ').title()}: {result['status']} {status_icon}")

        if results['issues']:
            print(f"\n🚨 Critical Issues ({len(results['issues'])}):")
            for issue in results['issues'][:5]:  # Show first 5
                print(f"  - {issue}")
            if len(results['issues']) > 5:
                print(f"  ... and {len(results['issues']) - 5} more")

        # Generate report
        report_file = verifier.generate_report(args.report)
        print(f"\n📋 Detailed report saved to: {report_file}")

    elif args.component:
        if args.component == 'database_schema':
            result = verifier.verify_database_schema()
        elif args.component == 'knowledge_models':
            result = verifier.verify_knowledge_models_setup()
        elif args.component == 'document_parsing':
            result = verifier.verify_document_parsing()
        elif args.component == 'embedding_index':
            result = verifier.verify_embedding_index()
        elif args.component == 'knowledge_api':
            if not args.quick:
                result = verifier.verify_knowledge_api()
            else:
                print("⏭️ Skipping API tests in quick mode")
                return
        elif args.component == 'quote_analysis':
            result = verifier.verify_quote_analysis()
        elif args.component == 'github_migration':
            result = verifier.verify_github_migration()
        elif args.component == 'frontend_interface':
            result = verifier.verify_frontend_interface()

        print(f"\n🔍 {args.component.replace('_', ' ').title()} Verification")
        print("=" * 50)
        print(f"Status: {result['status']}")

        if result['details']:
            print("Details:")
            for key, value in result['details'].items():
                print(f"  {key}: {value}")

        if result['issues']:
            print("Issues:")
            for issue in result['issues']:
                print(f"  - {issue}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()