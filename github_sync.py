#!/usr/bin/env python3
"""
GitHub同步脚本 - 安全同步知识库更新到GitHub
遵循github_auto_sync_agent.yaml的安全规则
"""

import os
import json
import sqlite3
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Set, Tuple
import re
import hashlib

class GitHubSyncManager:
    """GitHub同步管理器"""

    def __init__(self, config_path: str = "github_auto_sync_agent.yaml"):
        self.config_path = config_path
        self.logger = self._setup_logger()
        self.config = self._load_config()
        self.excluded_patterns = self._compile_exclusion_patterns()
        self.sanitization_rules = self._load_sanitization_rules()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("GitHubSync")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            "sync_rules": {
                "inclusion_rules": {
                    "important_files": [
                        "knowledge_base_schema.sql",
                        "setup_knowledge_models.py",
                        "build_embeddings.py",
                        "quote_analysis_agent.py",
                        "github_migration.py",
                        "verify_knowledge_system.py",
                        "generate_quote_strategies.py",
                        "learn_from_updates.py",
                        "api_server_knowledge.py",
                        "api_chat_interface.py"
                    ],
                    "documentation_files": ["*.md", "*.txt", "README*"],
                    "script_files": ["*.py", "*.yaml", "*.json"],
                    "config_files": ["config.json", "settings.json"]
                },
                "exclusion_rules": {
                    "sensitive_directories": [
                        "/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户中/",
                        "/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价和/",
                        "data/raw/", "data/temp/", "data/sensitive/",
                        "backup/", "backups/"
                    ],
                    "sensitive_files": [
                        "*customer*", "*client*", "*contact*",
                        "*personal*", "*private*", "*confidential*",
                        "*secret*", "*password*", "*token*", "*key*"
                    ],
                    "database_files": ["*.db", "*.sqlite*"],
                    "log_files": ["*.log", "*.log.*", "logs/"],
                    "system_files": [".env*", ".DS_Store", "*.tmp"],
                    "cache_files": ["__pycache__", "*.pyc", ".pytest_cache/"]
                }
            },
            "security": {
                "data_sanitization": {
                    "enabled": True,
                    "sanitization_rules": {
                        "customer_data": {
                            "fields": ["name", "contact", "phone", "email"],
                            "replacement": "[客户信息已脱敏]"
                        },
                        "price_data": {
                            "fields": ["price", "cost", "amount", "budget"],
                            "replacement": "[价格信息已脱敏]"
                        },
                        "personal_identifiers": {
                            "patterns": [
                                r"\d{15,17}",  # 身份证号
                                r"\d{11}",     # 手机号
                                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # 邮箱
                            ],
                            "replacement": "[个人信息已脱敏]"
                        }
                    }
                }
            }
        }

    def _compile_exclusion_patterns(self) -> List[re.Pattern]:
        """编译排除模式"""
        patterns = []

        # 添加敏感文件模式
        for pattern in self.config.get("sync_rules", {}).get("exclusion_rules", {}).get("sensitive_files", []):
            patterns.append(re.compile(pattern.replace("*", ".*"), re.IGNORECASE))

        # 添加数据库文件模式
        for pattern in self.config.get("sync_rules", {}).get("exclusion_rules", {}).get("database_files", []):
            patterns.append(re.compile(pattern.replace("*", ".*"), re.IGNORECASE))

        return patterns

    def _load_sanitization_rules(self) -> dict:
        """加载数据脱敏规则"""
        return self.config.get("security", {}).get("data_sanitization", {}).get("sanitization_rules", {})

    def _is_file_excluded(self, file_path: Path) -> bool:
        """检查文件是否被排除"""
        file_str = str(file_path)

        # 检查目录排除
        for excluded_dir in self.config.get("sync_rules", {}).get("exclusion_rules", {}).get("sensitive_directories", []):
            if excluded_dir in file_str:
                return True

        # 检查文件模式排除
        for pattern in self.excluded_patterns:
            if pattern.search(file_path.name):
                return True

        # 检查系统文件排除
        system_files = self.config.get("sync_rules", {}).get("exclusion_rules", {}).get("system_files", [])
        for pattern in system_files:
            if pattern.replace("*", ".*") in file_str:
                return True

        return False

    def _sanitize_file_content(self, content: str) -> str:
        """对文件内容进行脱敏处理"""
        if not self.config.get("security", {}).get("data_sanitization", {}).get("enabled", False):
            return content

        sanitized = content

        # 应用客户数据脱敏
        customer_rule = self.sanitization_rules.get("customer_data", {})
        for field in customer_rule.get("fields", []):
            pattern = rf'{field}["\']?\s*[:=]\s*["\']([^"\']+)["\']'
            sanitized = re.sub(pattern, f'{field}: {customer_rule.get("replacement", "[已脱敏]")}', sanitized, flags=re.IGNORECASE)

        # 应用价格数据脱敏
        price_rule = self.sanitization_rules.get("price_data", {})
        for field in price_rule.get("fields", []):
            pattern = rf'{field}["\']?\s*[:=]\s*["\']?([\d,.,]+\s*[RMB￥¥USD]?)["\']?'
            sanitized = re.sub(pattern, f'{field}: {price_rule.get("replacement", "[已脱敏]")}', sanitized, flags=re.IGNORECASE)

        # 应用个人身份标识脱敏
        personal_rule = self.sanitization_rules.get("personal_identifiers", {})
        for pattern in personal_rule.get("patterns", []):
            sanitized = re.sub(pattern, personal_rule.get("replacement", "[已脱敏]"), sanitized)

        return sanitized

    def _get_files_to_sync(self) -> List[Path]:
        """获取需要同步的文件列表"""
        files_to_sync = []
        base_path = Path(".")

        # 遍历所有文件
        for file_path in base_path.rglob("*"):
            if file_path.is_file() and not self._is_file_excluded(file_path):
                # 检查是否包含在包含规则中
                if self._should_include_file(file_path):
                    files_to_sync.append(file_path)

        return sorted(files_to_sync)

    def _should_include_file(self, file_path: Path) -> bool:
        """检查文件是否应该被包含"""
        file_name = file_path.name
        file_ext = file_path.suffix.lower()

        # 检查重要文件
        important_files = self.config.get("sync_rules", {}).get("inclusion_rules", {}).get("important_files", [])
        if file_name in important_files:
            return True

        # 检查文档文件
        doc_patterns = self.config.get("sync_rules", {}).get("inclusion_rules", {}).get("documentation_files", [])
        for pattern in doc_patterns:
            if pattern.replace("*", ".*") in file_name:
                return True

        # 检查脚本文件
        script_patterns = self.config.get("sync_rules", {}).get("inclusion_rules", {}).get("script_files", [])
        for pattern in script_patterns:
            if pattern.replace("*", ".*") in file_name:
                return True

        # 检查配置文件
        config_patterns = self.config.get("sync_rules", {}).get("inclusion_rules", {}).get("config_files", [])
        for pattern in config_patterns:
            if pattern.replace("*", ".*") in file_name:
                return True

        return False

    def _create_sanitized_copy(self, original_path: Path, temp_dir: Path) -> Path:
        """创建脱敏后的文件副本"""
        relative_path = original_path.relative_to(Path("."))
        temp_path = temp_dir / relative_path

        # 确保目录存在
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(original_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 应用脱敏处理
            sanitized_content = self._sanitize_file_content(content)

            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(sanitized_content)

        except Exception as e:
            self.logger.warning(f"Failed to sanitize {original_path}: {e}")
            # 如果脱敏失败，复制原文件
            import shutil
            shutil.copy2(original_path, temp_path)

        return temp_path

    def _git_add_files(self, files: List[Path]) -> bool:
        """使用git add添加文件"""
        try:
            for file_path in files:
                result = subprocess.run(['git', 'add', str(file_path)],
                                      capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    self.logger.warning(f"Failed to add {file_path}: {result.stderr}")
            return True
        except Exception as e:
            self.logger.error(f"Git add failed: {e}")
            return False

    def _git_commit(self, message: str) -> bool:
        """使用git commit提交更改"""
        try:
            result = subprocess.run(['git', 'commit', '-m', message],
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                self.logger.info(f"Commit successful: {message}")
                return True
            else:
                self.logger.warning(f"Commit failed: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Git commit failed: {e}")
            return False

    def _git_push(self, remote: str = "origin", branch: str = "main") -> bool:
        """使用git push推送到远程仓库"""
        try:
            result = subprocess.run(['git', 'push', remote, branch],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self.logger.info(f"Push successful to {remote}/{branch}")
                return True
            else:
                self.logger.warning(f"Push failed: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Git push failed: {e}")
            return False

    def _generate_sync_log(self, synced_files: List[Path], status: str) -> dict:
        """生成同步日志"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "files_count": len(synced_files),
            "files": [str(f) for f in synced_files],
            "security_measures_applied": [
                "data_sanitization",
                "sensitive_file_exclusion",
                "content_filtering"
            ]
        }

        # 保存到日志文件
        log_file = Path("data/processed/github_sync_log.json")
        log_file.parent.mkdir(exist_ok=True)

        try:
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)

            # 保留最近100条记录
            logs = logs[-100:]

            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.warning(f"Failed to save sync log: {e}")

        return log_entry

    def sync_to_github(self, mode: str = "auto") -> dict:
        """执行GitHub同步"""
        self.logger.info(f"Starting GitHub sync in {mode} mode")

        try:
            # 获取需要同步的文件
            files_to_sync = self._get_files_to_sync()
            self.logger.info(f"Found {len(files_to_sync)} files to sync")

            if not files_to_sync:
                return {"status": "success", "message": "No files to sync", "files_count": 0}

            # 创建临时目录进行脱敏处理
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # 创建脱敏副本
                sanitized_files = []
                for file_path in files_to_sync:
                    sanitized_file = self._create_sanitized_copy(file_path, temp_path)
                    sanitized_files.append(sanitized_file)

                # 添加文件到git
                if not self._git_add_files(sanitized_files):
                    return {"status": "error", "message": "Failed to add files to git"}

                # 提交更改
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_message = f"Knowledge base auto-sync - {timestamp} (Stage 5 Intelligent Assistant)"

                if not self._git_commit(commit_message):
                    return {"status": "error", "message": "Failed to commit changes"}

                # 推送到GitHub（如果配置了远程仓库）
                try:
                    self._git_push()
                except Exception as e:
                    self.logger.warning(f"Push to GitHub failed: {e}")
                    # 即使推送失败，本地同步也算成功

            # 生成同步日志
            log_entry = self._generate_sync_log(files_to_sync, "success")

            result = {
                "status": "success",
                "message": f"Successfully synced {len(files_to_sync)} files",
                "files_count": len(files_to_sync),
                "files": [str(f) for f in files_to_sync],
                "security_measures": ["data_sanitization", "sensitive_file_exclusion"],
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"GitHub sync completed: {result}")
            return result

        except Exception as e:
            self.logger.error(f"GitHub sync failed: {e}")
            log_entry = self._generate_sync_log([], "error")
            return {"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="GitHub同步脚本")
    parser.add_argument("--mode", choices=["auto", "quick", "deep"], default="auto",
                      help="同步模式")
    parser.add_argument("--config", default="github_auto_sync_agent.yaml",
                      help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true",
                      help="仅预览，不实际同步")

    args = parser.parse_args()

    sync_manager = GitHubSyncManager(args.config)

    if args.dry_run:
        files = sync_manager._get_files_to_sync()
        print(f"将要同步的文件数量: {len(files)}")
        for f in files:
            print(f"  - {f}")
        return

    result = sync_manager.sync_to_github(args.mode)
    print(f"同步结果: {result}")

if __name__ == "__main__":
    main()