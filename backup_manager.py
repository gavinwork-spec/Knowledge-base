#!/usr/bin/env python3
"""
备份管理脚本
提供自动备份、版本控制和恢复功能
"""

import os
import shutil
import sqlite3
import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import logging
import gzip
from typing import Dict

class BackupManager:
    """备份管理器"""

    def __init__(self, project_path: str = "./"):
        self.project_path = Path(project_path).resolve()
        self.db_path = self.project_path / "data" / "db.sqlite"
        self.backup_dir = self.project_path / "data" / "backups"
        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        log_dir = self.project_path / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'backup.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BackupManager')

    def create_database_backup(self, backup_type: str = "manual") -> str:
        """创建数据库备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"db_backup_{backup_type}_{timestamp}.sqlite"

        try:
            if self.db_path.exists():
                shutil.copy2(self.db_path, backup_file)
                self.logger.info(f"✅ 数据库备份完成: {backup_file}")
                return str(backup_file)
            else:
                self.logger.warning("⚠️ 数据库文件不存在")
                return ""

        except Exception as e:
            self.logger.error(f"❌ 数据库备份失败: {e}")
            return ""

    def create_compressed_backup(self) -> str:
        """创建压缩备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"project_backup_{timestamp}.tar.gz"

        try:
            # 使用tar创建压缩备份
            cmd = [
                'tar', '-czf', str(backup_file),
                '--exclude', '.git',
                '--exclude', 'logs',
                '--exclude', 'data/processed',
                '--exclude', 'data/failed',
                '--exclude', '__pycache__',
                '--exclude', '*.pyc',
                '--exclude', 'data/backups',
                '.'
            ]

            result = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"✅ 压缩备份完成: {backup_file}")
                return str(backup_file)
            else:
                self.logger.error(f"❌ 压缩备份失败: {result.stderr}")
                return ""

        except Exception as e:
            self.logger.error(f"❌ 压缩备份失败: {e}")
            return ""

    def create_git_snapshot(self, message: str = None) -> bool:
        """创建Git快照"""
        try:
            os.chdir(self.project_path)

            # 检查是否有Git仓库
            if not (self.project_path / ".git").exists():
                self.logger.warning("⚠️ Git仓库不存在，跳过Git快照")
                return False

            # 添加所有更改
            subprocess.run(['git', 'add', '.'], check=True, capture_output=True)

            # 创建提交
            if message is None:
                message = f"自动备份快照 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            result = subprocess.run(['git', 'commit', '-m', message],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"✅ Git快照完成: {message}")
                return True
            else:
                self.logger.warning(f"⚠️ Git快照无更改或失败: {result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Git快照失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Git快照异常: {e}")
            return False

    def cleanup_old_backups(self, days_to_keep: int = 30) -> Dict[str, int]:
        """清理旧备份"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        stats = {
            'deleted_db_backups': 0,
            'deleted_compressed_backups': 0,
            'freed_space_mb': 0
        }

        try:
            # 清理数据库备份
            for backup_file in self.backup_dir.glob("db_backup_*.sqlite"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    file_size = backup_file.stat().st_size / (1024 * 1024)  # MB
                    backup_file.unlink()
                    stats['deleted_db_backups'] += 1
                    stats['freed_space_mb'] += file_size
                    self.logger.info(f"🗑️ 删除旧备份: {backup_file.name}")

            # 清理压缩备份
            for backup_file in self.backup_dir.glob("project_backup_*.tar.gz"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    file_size = backup_file.stat().st_size / (1024 * 1024)  # MB
                    backup_file.unlink()
                    stats['deleted_compressed_backups'] += 1
                    stats['freed_space_mb'] += file_size
                    self.logger.info(f"🗑️ 删除旧压缩备份: {backup_file.name}")

            self.logger.info(f"✅ 备份清理完成，释放空间: {stats['freed_space_mb']:.1f} MB")
            return stats

        except Exception as e:
            self.logger.error(f"❌ 备份清理失败: {e}")
            return stats

    def list_backups(self) -> Dict[str, list]:
        """列出所有备份"""
        backups = {
            'database_backups': [],
            'compressed_backups': []
        }

        try:
            # 数据库备份
            for backup_file in sorted(self.backup_dir.glob("db_backup_*.sqlite"),
                                    key=lambda x: x.stat().st_mtime, reverse=True):
                backups['database_backups'].append({
                    'name': backup_file.name,
                    'path': str(backup_file),
                    'size_mb': backup_file.stat().st_size / (1024 * 1024),
                    'created_date': datetime.fromtimestamp(backup_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })

            # 压缩备份
            for backup_file in sorted(self.backup_dir.glob("project_backup_*.tar.gz"),
                                    key=lambda x: x.stat().st_mtime, reverse=True):
                backups['compressed_backups'].append({
                    'name': backup_file.name,
                    'path': str(backup_file),
                    'size_mb': backup_file.stat().st_size / (1024 * 1024),
                    'created_date': datetime.fromtimestamp(backup_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })

        except Exception as e:
            self.logger.error(f"❌ 列出备份失败: {e}")

        return backups

    def restore_database(self, backup_file: str) -> bool:
        """恢复数据库"""
        backup_path = Path(backup_file)

        if not backup_path.exists():
            self.logger.error(f"❌ 备份文件不存在: {backup_file}")
            return False

        try:
            # 创建当前数据库的备份
            current_backup = self.create_database_backup("before_restore")
            if current_backup:
                self.logger.info(f"✅ 当前数据库已备份: {current_backup}")

            # 恢复数据库
            shutil.copy2(backup_path, self.db_path)
            self.logger.info(f"✅ 数据库恢复完成: {backup_file}")

            # 验证恢复的数据库
            if self.verify_database():
                self.logger.info("✅ 数据库验证通过")
                return True
            else:
                self.logger.error("❌ 数据库验证失败")
                return False

        except Exception as e:
            self.logger.error(f"❌ 数据库恢复失败: {e}")
            return False

    def verify_database(self) -> bool:
        """验证数据库完整性"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 检查完整性
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()

            if result[0] == "ok":
                # 检查表是否存在
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                tables = [row[0] for row in cursor.fetchall()]

                expected_tables = ['customers', 'drawings', 'factories', 'factory_quotes', 'specifications', 'process_status']
                missing_tables = set(expected_tables) - set(tables)

                if missing_tables:
                    self.logger.error(f"❌ 缺失表: {missing_tables}")
                    return False

                conn.close()
                return True
            else:
                self.logger.error(f"❌ 数据库完整性检查失败: {result[0]}")
                return False

        except Exception as e:
            self.logger.error(f"❌ 数据库验证失败: {e}")
            return False

    def backup_metadata(self, backup_type: str = "manual") -> Dict[str, any]:
        """备份元数据信息"""
        try:
            metadata = {
                'backup_time': datetime.now().isoformat(),
                'backup_type': backup_type,
                'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
                'project_path': str(self.project_path),
                'git_status': self.get_git_status(),
                'database_stats': self.get_database_stats()
            }

            # 保存元数据
            metadata_file = self.backup_dir / f"backup_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return metadata

        except Exception as e:
            self.logger.error(f"❌ 备份元数据失败: {e}")
            return {}

    def get_git_status(self) -> Dict[str, any]:
        """获取Git状态"""
        try:
            os.chdir(self.project_path)

            # 检查是否有未提交的更改
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                changed_files = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                return {
                    'has_uncommitted_changes': changed_files > 0,
                    'changed_files_count': changed_files
                }
            else:
                return {'error': 'Git status failed'}

        except Exception as e:
            return {'error': str(e)}

    def get_database_stats(self) -> Dict[str, int]:
        """获取数据库统计"""
        if not self.db_path.exists():
            return {}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            stats = {}
            tables = ['customers', 'drawings', 'factories', 'factory_quotes', 'specifications', 'process_status']

            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]
                except:
                    stats[table] = 0

            conn.close()
            return stats

        except Exception:
            return {}

    def run_full_backup(self, message: str = None) -> Dict[str, str]:
        """运行完整备份流程"""
        self.logger.info("🚀 开始完整备份流程...")

        results = {
            'database_backup': '',
            'compressed_backup': '',
            'git_snapshot': '',
            'metadata': ''
        }

        # 1. 数据库备份
        db_backup = self.create_database_backup("scheduled")
        results['database_backup'] = db_backup

        # 2. Git快照
        git_success = self.create_git_snapshot(message)
        results['git_snapshot'] = 'success' if git_success else 'failed'

        # 3. 压缩备份
        compressed_backup = self.create_compressed_backup()
        results['compressed_backup'] = compressed_backup

        # 4. 备份元数据
        metadata = self.backup_metadata("scheduled")
        results['metadata'] = 'success' if metadata else 'failed'

        # 5. 清理旧备份
        cleanup_stats = self.cleanup_old_backups()
        results['cleanup'] = f"删除 {cleanup_stats['deleted_db_backups']} 个DB备份, {cleanup_stats['deleted_compressed_backups']} 个压缩备份"

        self.logger.info("✅ 完整备份流程完成")
        return results

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='备份管理工具')
    parser.add_argument('--type', choices=['db', 'compressed', 'git', 'full'],
                       default='full', help='备份类型')
    parser.add_argument('--message', help='Git提交消息')
    parser.add_argument('--list', action='store_true', help='列出备份')
    parser.add_argument('--cleanup', type=int, help='清理N天前的备份')
    parser.add_argument('--restore', help='恢复数据库备份')

    args = parser.parse_args()

    manager = BackupManager()

    if args.list:
        backups = manager.list_backups()
        print("📋 数据库备份:")
        for backup in backups['database_backups'][:5]:
            print(f"  {backup['created_date']} - {backup['name']} ({backup['size_mb']:.1f} MB)")

        print("\n📦 压缩备份:")
        for backup in backups['compressed_backups'][:3]:
            print(f"  {backup['created_date']} - {backup['name']} ({backup['size_mb']:.1f} MB)")

    elif args.restore:
        success = manager.restore_database(args.restore)
        if success:
            print("✅ 数据库恢复成功")
        else:
            print("❌ 数据库恢复失败")

    elif args.cleanup:
        stats = manager.cleanup_old_backups(args.cleanup)
        print(f"✅ 清理完成: 删除 {stats['deleted_db_backups']} 个DB备份, "
              f"{stats['deleted_compressed_backups']} 个压缩备份, "
              f"释放 {stats['freed_space_mb']:.1f} MB 空间")

    else:
        if args.type == 'db':
            manager.create_database_backup()
        elif args.type == 'compressed':
            manager.create_compressed_backup()
        elif args.type == 'git':
            manager.create_git_snapshot(args.message)
        else:  # full
            results = manager.run_full_backup(args.message)
            print("✅ 完整备份完成:")
            for backup_type, result in results.items():
                print(f"  {backup_type}: {result}")

if __name__ == "__main__":
    main()