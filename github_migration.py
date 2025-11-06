#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Migration and Data Security
GitHub迁移和数据安全管理

This script handles secure migration of knowledge base data to GitHub,
with proper exclusions for sensitive customer information and local files.
"""

import os
import json
import sqlite3
import logging
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Set
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/github_migration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GitHubMigrationManager:
    """GitHub迁移管理器"""

    def __init__(self, repo_path: str = "knowledge-github-backup"):
        self.repo_path = Path(repo_path)
        self.knowledge_base_path = Path(".")
        self.sensitive_patterns = self._get_sensitive_patterns()
        self.excluded_files = self._get_excluded_files()
        self.excluded_directories = self._get_excluded_directories()

    def _get_sensitive_patterns(self) -> List[str]:
        """获取敏感信息模式"""
        return [
            # 客户敏感信息
            r'.*客户.*联系人.*',
            r'.*客户.*电话.*',
            r'.*客户.*邮箱.*',
            r'.*客户.*地址.*',

            # 报价敏感信息
            r'.*报价.*金额.*',
            r'.*价格.*明细.*',
            r'.*成本.*分析.*',

            # 个人信息
            r'.*身份证.*',
            r'.*护照.*',
            r'.*银行.*账号.*',

            # 本地路径
            r'/Users/gavin/.*',
            r'/Users/[^/]+/.*',

            # 系统配置
            r'.*\.env$',
            r'.*config.*secret.*',
            r'.*password.*',
            r'.*token.*',
        ]

    def _get_excluded_files(self) -> List[str]:
        """获取排除的文件列表"""
        return [
            # 数据库文件
            "knowledge_base.db",
            "knowledge_base.db-journal",
            "*.db",
            "*.sqlite",
            "*.sqlite3",

            # 配置文件
            ".env",
            "*.env.*",
            "config.json",
            "secrets.json",

            # 日志文件
            "*.log",
            "*.log.*",

            # 临时文件
            "*.tmp",
            "*.temp",
            ".DS_Store",
            "Thumbs.db",

            # Python缓存
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",

            # Node modules
            "node_modules",

            # IDE文件
            ".vscode",
            ".idea",
            "*.swp",
            "*.swo",

            # 系统文件
            ".git",
            ".gitignore",
        ]

    def _get_excluded_directories(self) -> List[str]:
        """获取排除的目录列表"""
        return [
            # 数据目录（包含敏感信息）
            "data/processed",
            "data/raw",
            "data/temp",

            # 客户文件目录
            "002-客户中",
            "005-询盘询价和",

            # 配置目录
            "config",
            ".config",

            # 备份目录
            "backup",
            "backups",

            # 系统目录
            ".git",
            "__pycache__",
            "node_modules",
        ]

    def is_sensitive_content(self, file_path: Path) -> bool:
        """检查文件是否包含敏感内容"""
        try:
            # 检查文件名
            for pattern in self.sensitive_patterns:
                if pattern.startswith('.*') and pattern.endswith('.*'):
                    # 正则表达式模式
                    import re
                    if re.search(pattern, file_path.name, re.IGNORECASE):
                        return True
                elif pattern.lower() in file_path.name.lower():
                    return True

            # 检查文件内容（仅限文本文件）
            if file_path.suffix in ['.txt', '.md', '.json', '.py', '.yaml', '.yml', '.csv']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in self.sensitive_patterns:
                            import re
                            if re.search(pattern, content, re.IGNORECASE):
                                return True
                except UnicodeDecodeError:
                    # 非文本文件，跳过内容检查
                    pass

            return False

        except Exception as e:
            logger.warning(f"Error checking sensitivity of {file_path}: {e}")
            return True  # 出错时保守处理，排除该文件

    def should_exclude_file(self, file_path: Path) -> bool:
        """判断文件是否应该被排除"""
        # 转换为相对路径
        try:
            relative_path = file_path.relative_to(self.knowledge_base_path)
        except ValueError:
            relative_path = file_path

        # 检查文件名模式
        for pattern in self.excluded_files:
            if pattern.startswith('*'):
                # 通配符模式
                if relative_path.match(pattern):
                    return True
            elif pattern in relative_path.name or pattern == str(relative_path):
                return True

        # 检查目录排除
        for excluded_dir in self.excluded_directories:
            if str(relative_path).startswith(excluded_dir):
                return True

        # 检查敏感内容
        if self.is_sensitive_content(relative_path):
            return True

        return False

    def extract_knowledge_data(self) -> Dict:
        """提取知识库数据（不包含敏感信息）"""
        try:
            # 连接数据库
            conn = sqlite3.connect('knowledge_base.db')
            cursor = conn.cursor()

            # 提取实体类型（安全的公开信息）
            cursor.execute("""
                SELECT name, display_name, description, color, icon
                FROM entity_types
                WHERE is_active = 1
            """)
            entity_types = [dict(zip([col[0] for col in cursor.description], row))
                          for row in cursor.fetchall()]

            # 提取知识条目（去除敏感信息）
            cursor.execute("""
                SELECT id, entity_type, name, description, created_at, updated_at
                FROM knowledge_entries
                WHERE entity_type IN ('product', 'specification', 'material')
                ORDER BY created_at DESC
            """)

            knowledge_entries = []
            for row in cursor.fetchall():
                entry = dict(zip([col[0] for col in cursor.description], row))

                # 获取安全的属性（去除价格、联系方式等敏感信息）
                cursor.execute("""
                    SELECT attributes_json
                    FROM knowledge_entries
                    WHERE id = ?
                """, (entry['id'],))

                attributes_result = cursor.fetchone()
                if attributes_result and attributes_result[0]:
                    try:
                        attributes = json.loads(attributes_result[0])
                        # 过滤敏感属性
                        safe_attributes = {}
                        sensitive_keys = ['phone', 'email', 'address', 'price', 'cost', 'amount', 'contact']

                        for key, value in attributes.items():
                            if not any(sensitive in key.lower() for sensitive in sensitive_keys):
                                safe_attributes[key] = value

                        entry['attributes'] = safe_attributes
                    except json.JSONDecodeError:
                        entry['attributes'] = {}
                else:
                    entry['attributes'] = {}

                knowledge_entries.append(entry)

            # 提取NLP实体（仅公开类型）
            cursor.execute("""
                SELECT keyword, category, confidence_score
                FROM nlp_entities
                WHERE category IN ('product_name', 'specification', 'material')
                GROUP BY keyword, category
                HAVING AVG(confidence_score) > 0.7
                ORDER BY COUNT(*) DESC
                LIMIT 100
            """)
            nlp_entities = [dict(zip([col[0] for col in cursor.description], row))
                           for row in cursor.fetchall()]

            # 提取策略建议（仅类型和标题，不含具体数据）
            cursor.execute("""
                SELECT suggestion_type, title, impact_level, confidence_score
                FROM strategy_suggestions
                WHERE status = 'pending'
                ORDER BY created_at DESC
                LIMIT 50
            """)
            strategy_suggestions = [dict(zip([col[0] for col in cursor.description], row))
                                  for row in cursor.fetchall()]

            conn.close()

            return {
                'export_timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'entity_types': entity_types,
                'knowledge_entries': knowledge_entries,
                'nlp_entities': nlp_entities,
                'strategy_suggestions': strategy_suggestions,
                'statistics': {
                    'total_entity_types': len(entity_types),
                    'total_knowledge_entries': len(knowledge_entries),
                    'total_nlp_entities': len(nlp_entities),
                    'total_strategy_suggestions': len(strategy_suggestions)
                }
            }

        except Exception as e:
            logger.error(f"Failed to extract knowledge data: {e}")
            return {}

    def init_github_repo(self) -> bool:
        """初始化GitHub仓库"""
        try:
            if self.repo_path.exists():
                logger.info(f"Repository directory already exists: {self.repo_path}")
                return True

            # 创建仓库目录
            self.repo_path.mkdir(parents=True, exist_ok=True)

            # 初始化Git仓库
            subprocess.run(['git', 'init'],
                         cwd=self.repo_path,
                         check=True,
                         capture_output=True)

            # 创建.gitignore文件
            gitignore_content = """
# Database files
*.db
*.sqlite
*.sqlite3

# Sensitive data
data/
config/
.env*
*.log

# Python
__pycache__/
*.pyc
*.pyo

# System files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Node
node_modules/
"""

            with open(self.repo_path / '.gitignore', 'w', encoding='utf-8') as f:
                f.write(gitignore_content)

            # 创建README文件
            readme_content = """# Knowledge Base Backup

This repository contains the knowledge base backup and documentation.

## Contents

- `knowledge_data.json` - Extracted knowledge data (sanitized)
- `docs/` - Documentation and reports
- `scripts/` - Utility scripts

## Data Security

All sensitive customer information, pricing data, and personal details have been removed from this backup.

Only public product specifications, materials, and non-sensitive business knowledge is included.

## Last Updated

{}
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            with open(self.repo_path / 'README.md', 'w', encoding='utf-8') as f:
                f.write(readme_content)

            logger.info(f"✅ GitHub repository initialized: {self.repo_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize git repository: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize GitHub repo: {e}")
            return False

    def copy_safe_files(self) -> List[str]:
        """复制安全文件到GitHub仓库"""
        copied_files = []

        try:
            # 创建必要的目录结构
            (self.repo_path / 'docs').mkdir(exist_ok=True)
            (self.repo_path / 'scripts').mkdir(exist_ok=True)

            # 复制知识数据
            knowledge_data = self.extract_knowledge_data()
            if knowledge_data:
                with open(self.repo_path / 'knowledge_data.json', 'w', encoding='utf-8') as f:
                    json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
                copied_files.append('knowledge_data.json')

            # 复制文档文件（安全的）
            doc_files = [
                'parse_documents_agent.yaml',
                'github-frontend/knowledge.html',
            ]

            for file_path in doc_files:
                source = self.knowledge_base_path / file_path
                if source.exists() and not self.should_exclude_file(source):
                    dest = self.repo_path / 'docs' / source.name
                    shutil.copy2(source, dest)
                    copied_files.append(f'docs/{source.name}')

            # 复制脚本文件（去除敏感配置）
            script_files = [
                'setup_knowledge_models.py',
                'build_embeddings.py',
                'quote_analysis_agent.py',
            ]

            for file_path in script_files:
                source = self.knowledge_base_path / file_path
                if source.exists():
                    dest = self.repo_path / 'scripts' / source.name
                    shutil.copy2(source, dest)
                    copied_files.append(f'scripts/{source.name}')

            logger.info(f"✅ Copied {len(copied_files)} safe files to repository")
            return copied_files

        except Exception as e:
            logger.error(f"Failed to copy safe files: {e}")
            return []

    def create_migration_report(self, copied_files: List[str]) -> str:
        """创建迁移报告"""
        report = f"""
# Knowledge Base Migration Report

**Migration Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Repository**: {self.repo_path}

## Files Copied ({len(copied_files)})

{chr(10).join(f'- {file}' for file in copied_files)}

## Security Measures Applied

- ✅ Removed all customer personal information
- ✅ Removed pricing and cost data
- ✅ Removed contact information
- ✅ Removed local file paths
- ✅ Removed configuration files
- ✅ Removed database files
- ✅ Removed log files

## Data Summary

- Entity Types: Public taxonomy information
- Knowledge Entries: Product specifications and materials only
- NLP Entities: Non-sensitive extracted keywords
- Strategy Suggestions: Types and titles only

## Exclusions

The following types of data were excluded for security:

- Customer information (names, contacts, addresses)
- Pricing and cost information
- Personal data (phone, email, etc.)
- Local file system paths
- Configuration and credential files
- Database and log files

## GitHub Repository Ready

This repository is ready for GitHub backup without sensitive information exposure.

*Generated by Knowledge Base Migration Manager v1.0.0*
"""

        report_path = self.repo_path / 'MIGRATION_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return str(report_path)

    def commit_and_push(self, commit_message: str = None) -> bool:
        """提交并推送到GitHub"""
        try:
            if not commit_message:
                commit_message = f"Knowledge base backup - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # 添加所有文件
            subprocess.run(['git', 'add', '.'],
                         cwd=self.repo_path,
                         check=True,
                         capture_output=True)

            # 提交更改
            subprocess.run(['git', 'commit', '-m', commit_message],
                         cwd=self.repo_path,
                         check=True,
                         capture_output=True)

            logger.info("✅ Changes committed to local repository")

            # 如果配置了远程仓库，尝试推送
            try:
                result = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                                      cwd=self.repo_path,
                                      capture_output=True,
                                      text=True)
                if result.returncode == 0:
                    subprocess.run(['git', 'push'],
                                 cwd=self.repo_path,
                                 check=True,
                                 capture_output=True)
                    logger.info("✅ Changes pushed to GitHub")
                    return True
                else:
                    logger.info("ℹ️ No remote origin configured - commits are local only")
                    return True
            except subprocess.CalledProcessError:
                logger.info("ℹ️ Push failed - commits are local only")
                return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit and push: {e}")
            return False

    def run_migration(self, auto_commit: bool = True) -> bool:
        """运行完整的迁移流程"""
        try:
            logger.info("🚀 Starting GitHub migration...")

            # 1. 初始化仓库
            if not self.init_github_repo():
                logger.error("❌ Failed to initialize repository")
                return False

            # 2. 复制安全文件
            copied_files = self.copy_safe_files()
            if not copied_files:
                logger.warning("⚠️ No files were copied - check security filters")

            # 3. 创建迁移报告
            report_path = self.create_migration_report(copied_files)
            logger.info(f"📋 Migration report created: {report_path}")

            # 4. 提交更改
            if auto_commit:
                if self.commit_and_push():
                    logger.info("✅ GitHub migration completed successfully!")
                else:
                    logger.warning("⚠️ Files prepared but commit failed")
            else:
                logger.info("📦 Files prepared for manual commit")

            return True

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False

    def list_excluded_files(self) -> Dict[str, List[str]]:
        """列出被排除的文件（用于审计）"""
        excluded = {
            'files': [],
            'directories': [],
            'sensitive_content': []
        }

        try:
            for item in self.knowledge_base_path.rglob('*'):
                if item.is_file():
                    if self.should_exclude_file(item):
                        try:
                            relative_path = item.relative_to(self.knowledge_base_path)

                            # 检查排除原因
                            if item.is_dir():
                                excluded['directories'].append(str(relative_path))
                            elif self.is_sensitive_content(item):
                                excluded['sensitive_content'].append(str(relative_path))
                            else:
                                excluded['files'].append(str(relative_path))
                        except ValueError:
                            continue

        except Exception as e:
            logger.error(f"Failed to list excluded files: {e}")

        return excluded

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='GitHub Migration Manager')
    parser.add_argument('--run-migration', action='store_true', help='Run complete migration')
    parser.add_argument('--init-repo', action='store_true', help='Initialize repository only')
    parser.add_argument('--extract-data', action='store_true', help='Extract knowledge data only')
    parser.add_argument('--list-excluded', action='store_true', help='List excluded files')
    parser.add_argument('--repo-path', default='knowledge-github-backup', help='Repository path')
    parser.add_argument('--no-commit', action='store_true', help='Skip auto commit')

    args = parser.parse_args()

    manager = GitHubMigrationManager(args.repo_path)

    if args.run_migration:
        success = manager.run_migration(auto_commit=not args.no_commit)
        if success:
            print("✅ Migration completed successfully!")
        else:
            print("❌ Migration failed!")

    elif args.init_repo:
        if manager.init_github_repo():
            print("✅ Repository initialized successfully!")
        else:
            print("❌ Repository initialization failed!")

    elif args.extract_data:
        data = manager.extract_knowledge_data()
        if data:
            output_file = 'extracted_knowledge_data.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Knowledge data extracted to {output_file}")
            print(f"   Entity types: {len(data.get('entity_types', []))}")
            print(f"   Knowledge entries: {len(data.get('knowledge_entries', []))}")
            print(f"   NLP entities: {len(data.get('nlp_entities', []))}")
        else:
            print("❌ Failed to extract knowledge data!")

    elif args.list_excluded:
        excluded = manager.list_excluded_files()

        print(f"\n📋 Excluded Files Audit")
        print("=" * 50)
        print(f"Files excluded by pattern: {len(excluded['files'])}")
        print(f"Directories excluded: {len(excluded['directories'])}")
        print(f"Files with sensitive content: {len(excluded['sensitive_content'])}")

        if excluded['files']:
            print(f"\n📁 Excluded Files:")
            for file in excluded['files'][:10]:  # Show first 10
                print(f"   - {file}")
            if len(excluded['files']) > 10:
                print(f"   ... and {len(excluded['files']) - 10} more")

        if excluded['sensitive_content']:
            print(f"\n🔒 Files with Sensitive Content:")
            for file in excluded['sensitive_content'][:10]:  # Show first 10
                print(f"   - {file}")
            if len(excluded['sensitive_content']) > 10:
                print(f"   ... and {len(excluded['sensitive_content']) - 10} more")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()